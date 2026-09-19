"""Production-cache stages and the safe local live-orchestration seam."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from .cache import cached_stage, content_key, reserve, settle
from .dose_search import CALIBRATION_CASE, TRANSFER_CASES, fit_target, predict_transfer
from .sweep import BSBENCH_PERSONAS, BSBENCH_PERSONA_N_PAIRS, BSBENCH_PERSONA_SEED, BSBENCH_PERSONA_TEMPLATE, BSBENCH_PERSONA_THINKING, CANDIDATE_DOSE_UPPER, MODAL_GPU_STAGE_UPPER_USD, case_identity, final_stages
from .transfer_data import load_transfer_records, transfer_provenance
from .validation import numbered_requests, validate_persona_examples


def production_stage(root: Path, ledger: Path, *, stage: str, model: dict, data: dict, method: str, config: dict, prompts: list[str], backend, validate_result=None, dispatch_config=None) -> dict:
    """Dispatch only on a cache miss, after reserving its declared upper cost."""
    dispatched = False

    def compute() -> dict:
        nonlocal dispatched
        reservation = reserve(ledger, f"modal-{stage}-{method}", config["upper_usd"], limit_usd=50.0)
        dispatched = True
        backend_config = config if dispatch_config is None else dispatch_config(config)
        result = backend.gpu(stage=stage, method=method, config=backend_config, prompts=prompts)
        if validate_result:
            validate_result(result)
        settle(ledger, reservation, result["actual_usd"])
        return result | {"reservation": reservation}

    result = cached_stage(root / "cache", stage, model=model, data=data, method=method, config=config, prompts=prompts, compute=compute)
    return result | {"reused": not dispatched}


def record_completed_stage(root: Path, *, stage: str, model: dict, data: dict, method: str, config: dict, prompts: list[str], result: dict) -> dict:
    return cached_stage(root / "cache", stage, model=model, data=data, method=method, config=config, prompts=prompts, compute=lambda: result)


def persist_local_judge_work(root: Path, *, model: dict, data: dict, rows: list[dict], persona_examples: list[dict], endpoint: str) -> dict:
    prompts = [row["prompt"] for row in rows]
    return cached_stage(root / "cache", "local-judge-work", model=model, data=data, method="judge", config={"endpoint": endpoint, "persona_pairs": len(persona_examples)}, prompts=prompts, compute=lambda: {"persona_checks": validate_persona_examples(persona_examples), "requests": numbered_requests(rows, model["judge_model"], endpoint)})


def run_stages(root: Path, ledger: Path, stages: list[dict], backend) -> list[dict]:
    return [production_stage(root, ledger, backend=backend, **item) for item in stages]


def persona_source_identity() -> dict:
    return {"pairs": [list(pair) for pair in BSBENCH_PERSONAS], "template": BSBENCH_PERSONA_TEMPLATE, "seed": BSBENCH_PERSONA_SEED, "n_pairs": BSBENCH_PERSONA_N_PAIRS, "thinking": BSBENCH_PERSONA_THINKING}


def _sidecar(root: Path, payload: str | bytes) -> dict:
    raw = base64.b64decode(payload) if isinstance(payload, str) else payload
    if not raw:
        raise ValueError("calibration candidate returned an empty vector sidecar")
    digest = hashlib.sha256(raw).hexdigest()
    path = root / "artifacts" / "vectors" / f"{digest}.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError(f"vector sidecar collision at {path}")
    path.write_bytes(raw)
    return {"path": str(path.relative_to(root)), "sha256": digest, "bytes": len(raw), "format": "binary-sidecar-v1"}


def _load_sidecar(root: Path, artifact: dict) -> dict:
    path = root / artifact["path"]
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise RuntimeError("durable vector sidecar is missing or corrupt; refusing final dispatch")
    return artifact


def _dispatch_sidecar(root: Path, config: dict) -> dict:
    artifact = _load_sidecar(root, config["vector_artifact"])
    return config | {"vector_artifact": artifact | {"backend_path": str((root / artifact["path"]).resolve())}}


def _local(root: Path, *, stage: str, model: dict, data: dict, method: str, prompts: list[str], config: dict, compute) -> dict:
    return cached_stage(root / "cache", stage, model=model, data=data, method=method, config=config, prompts=prompts, compute=compute)


def _candidate_items(coefficients: list[float], prompts: list[str], items: list[dict]) -> list[dict]:
    """Require one executable candidate response for every coefficient/prompt pair."""
    required = {"coefficient", "prompt_index", "prompt_sha256", "response"}
    expected = {(float(coefficient), index, hashlib.sha256(prompt.encode()).hexdigest()) for coefficient in coefficients for index, prompt in enumerate(prompts)}
    actual = set()
    for item in items:
        if not required.issubset(item) or not isinstance(item["response"], str):
            raise ValueError("candidate items require coefficient, prompt index/hash and response")
        actual.add((float(item["coefficient"]), item["prompt_index"], item["prompt_sha256"]))
    if actual != expected or len(items) != len(expected):
        raise ValueError("candidate items must cover exactly every coefficient and calibration prompt once")
    return items


def _require_observed(rows: list[dict] | None, coefficients: list[float]) -> list[dict]:
    if not rows:
        raise ValueError("explicit candidate observations are required before target fitting")
    required = {"coefficient", "useful", "coherent", "provenance", "generation_health"}
    if any(not required.issubset(row) for row in rows):
        raise ValueError("candidate observations require coefficient, useful, coherent, provenance and generation_health")
    if {float(row["coefficient"]) for row in rows} != {float(coefficient) for coefficient in coefficients}:
        raise ValueError("observed coefficient set must exactly match candidate coefficients")
    return rows


def _plan(records: dict, dose_plans: list[dict]) -> list[dict]:
    """Expand the already-validated canonical final-dose plans into executable items."""
    by_case = {plan["case"]["case_id"]: plan for plan in dose_plans}
    return [
        {"case_id": case.case_id, "target_id": by_case[case.case_id]["target_id"], "coefficient": coefficient,
         "prompt_id": record.prompt_id, "prompt": record.prompt, "prompt_sha256": record.content_sha256}
        for case in TRANSFER_CASES for record in records[case.case_id]
        for coefficient in by_case[case.case_id]["coefficients"]
    ]


def _validate_final(plan: list[dict], result: dict) -> None:
    answers = result.get("answers")
    if not isinstance(answers, list) or len(answers) != len(plan):
        raise ValueError("final backend must return exactly one answer per planned item")
    if result.get("plan_sha256") != content_key({"plan": plan}):
        raise ValueError("final backend result does not attest to the executable generation plan")


def run_direct_condition(root: Path, ledger: Path, *, model: dict, data: dict, method: str, prompts: list[str], backend, prompt_spec: dict) -> dict:
    """Cached direct bare/prompting path; it intentionally has no vector calibration."""
    if method not in {"bare", "prompting"}:
        raise ValueError("direct conditions are bare or prompting")
    config = {"upper_usd": MODAL_GPU_STAGE_UPPER_USD, "prompt_spec": prompt_spec, "condition": method}
    generation = production_stage(root, ledger, stage="generation", model=model, data=data, method=method, config=config, prompts=prompts, backend=backend, validate_result=lambda result: len(result.get("answers", [])) == len(prompts) or (_ for _ in ()).throw(ValueError("direct backend must return one answer per prompt")))
    identity = {"generation_sha256": content_key({key: value for key, value in generation.items() if key != "reused"}), "fake": True}
    items = [{"prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response": answer, "fake": True} for prompt, answer in zip(prompts, generation["answers"], strict=True)]
    return {"paid_execution_enabled": False, "generation": generation,
            "health": _local(root, stage="generation-health", model=model, data=data, method=method, prompts=prompts, config=identity, compute=lambda: {"schema": "bsbench-local-health-v1", "fake": True, "records": items}),
            "aware": _local(root, stage="target-aware-requests", model=model, data=data, method=method, prompts=prompts, config=identity, compute=lambda: {"schema": "bsbench-local-aware-v1", "fake": True, "records": items}),
            "blind": _local(root, stage="blind-requests", model=model, data=data, method=method, prompts=prompts, config=identity, compute=lambda: {"schema": "bsbench-local-blind-v1", "fake": True, "records": items})}


def run_live_two_step(root: Path, ledger: Path, *, model: dict, data: dict, method: str, calibration_prompts: list[str], backend, prompt_spec: dict, candidate_judgments: list[dict] | None, measure, solver, vector_loader, transfer_records: dict | None = None, extraction_identity: dict | None = None) -> dict:
    """Execute the audited vector graph using real target/prediction functions and injected local measurement dependencies."""
    if method in {"bare", "prompting"} or len(calibration_prompts) != 4:
        raise ValueError("vector orchestration requires a vector method and exactly four calibration prompts")
    expected_identity = persona_source_identity()
    source = expected_identity if extraction_identity is None else extraction_identity
    # Tests may version an otherwise exact identity; production rejects any semantic persona change.
    if {key: source[key] for key in expected_identity} != expected_identity:
        raise ValueError("vector calibration requires the fixed sycophantic/abrasive persona identity")
    records = transfer_records if transfer_records is not None else load_transfer_records()
    calibration_config = {"upper_usd": MODAL_GPU_STAGE_UPPER_USD, "calibration_case": case_identity(CALIBRATION_CASE), "persona_source": source, "persona_source_sha256": content_key(source), "candidate_dose_upper": CANDIDATE_DOSE_UPPER, "prompt_spec": prompt_spec}
    dispatched = False
    def candidate_compute() -> dict:
        nonlocal dispatched
        reservation = reserve(ledger, f"modal-calibration-candidates-{method}", calibration_config["upper_usd"], limit_usd=50.0)
        dispatched = True
        result = backend.gpu(stage="calibration-candidates", method=method, config=calibration_config, prompts=calibration_prompts)
        if "vector_bytes" not in result:
            raise ValueError("calibration backend must return vector_bytes, not a container path")
        artifact = _sidecar(root, result.pop("vector_bytes"))
        coefficients = result.get("candidate_coefficients")
        if not coefficients or len(coefficients) > CANDIDATE_DOSE_UPPER or len({float(coefficient) for coefficient in coefficients}) != len(coefficients):
            raise ValueError(f"calibration backend must return 1..{CANDIDATE_DOSE_UPPER} unique candidate coefficients")
        items = _candidate_items(coefficients, calibration_prompts, result.get("candidate_items", []))
        settle(ledger, reservation, result["actual_usd"])
        return result | {"candidate_items": items, "reservation": reservation, "vector_artifact": artifact, "vector_sha256": artifact["sha256"], "method_config": result.get("method_config", {})}
    candidate = cached_stage(root / "cache", "calibration-candidates", model=model, data=data, method=method, config=calibration_config, prompts=calibration_prompts, compute=candidate_compute)
    artifact = _load_sidecar(root, candidate["vector_artifact"])
    candidate = candidate | {"reused": not dispatched}
    candidate_identity = {key: value for key, value in candidate.items() if key != "reused"}
    observed = _require_observed(candidate_judgments, candidate["candidate_coefficients"])
    candidate_inputs = {"candidate_sha256": content_key(candidate_identity), "observed": observed, "vector_sha256": candidate["vector_sha256"]}
    health = _local(root, stage="candidate-health", model=model, data=data, method=method, prompts=calibration_prompts, config=candidate_inputs, compute=lambda: {"schema": "bsbench-local-health-v1", "fake": True, "candidate_items": candidate["candidate_items"], "records": observed})
    aware = _local(root, stage="candidate-aware", model=model, data=data, method=method, prompts=calibration_prompts, config=candidate_inputs, compute=lambda: {"schema": "bsbench-local-aware-v1", "fake": True, "candidate_items": candidate["candidate_items"], "records": observed})
    blind = _local(root, stage="candidate-blind", model=model, data=data, method=method, prompts=calibration_prompts, config=candidate_inputs, compute=lambda: {"schema": "bsbench-local-blind-v1", "fake": True, "candidate_items": candidate["candidate_items"], "records": observed})
    vector = vector_loader(artifact)
    target_config = {"candidate_sha256": content_key(candidate_identity), "observed_sha256": content_key({"observed": observed}), "vector_sha256": candidate["vector_sha256"], "candidate_records": [health, aware, blind]}
    target = _local(root, stage="fit-target", model=model, data=data, method=method, prompts=calibration_prompts, config=target_config, compute=lambda: fit_target(vector, model, None, calibration_prompts, CALIBRATION_CASE, observed, method=method, model_id=model["id"], measure_kwargs={}, measure=measure) | {"extraction_identity": source})
    provenance = {case_id: transfer_provenance(case_records) for case_id, case_records in records.items()}
    transfer_prompts = [record.prompt for case_records in records.values() for record in case_records]
    prediction_config = {"target": target, "transfer_provenance": provenance, "prompt_spec": prompt_spec, "vector_sha256": candidate["vector_sha256"]}
    prediction = _local(root, stage="transfer-prediction", model=model, data=data, method=method, prompts=transfer_prompts, config=prediction_config, compute=lambda: {"predictions": [predict_transfer(vector, model, None, [record.prompt for record in records[case.case_id]], target, case, bracket=(0.01, 2.0), solver_kwargs={}, solver=solver) for case in TRANSFER_CASES]})
    stages = final_stages(method=method, vector_sha256=candidate["vector_sha256"], observed=observed, transfer_predictions=prediction["predictions"], case_prompts=records, prompt_spec=prompt_spec)
    executable_plan = _plan(records, stages[0]["config"]["final_dose_plans"])
    plan_prompts = [json.dumps(item, sort_keys=True) for item in executable_plan]
    final_config = stages[0]["config"] | {"upper_usd": MODAL_GPU_STAGE_UPPER_USD, "extraction_identity": source, "vector_artifact": artifact, "executable_generation_plan": executable_plan, "executable_plan_sha256": content_key({"plan": executable_plan})}
    final = production_stage(root, ledger, stage="final-generation", model=model, data=data, method=method, config=final_config, prompts=plan_prompts, backend=backend, validate_result=lambda result: _validate_final(executable_plan, result), dispatch_config=lambda config: _dispatch_sidecar(root, config))
    final_inputs = {"final_sha256": content_key({key: value for key, value in final.items() if key != "reused"}), "plan": executable_plan, "target": target}
    fake_records = [{**item, "response": answer, "fake": True, "non_experimental": True} for item, answer in zip(executable_plan, final["answers"], strict=True)]
    return {"paid_execution_enabled": False, "candidate": candidate, "candidate_health": health, "candidate_aware": aware, "candidate_blind": blind, "target": target, "transfer_prediction": prediction, "final_stages": stages, "final": final,
            "final_health": _local(root, stage="final-health", model=model, data=data, method=method, prompts=plan_prompts, config=final_inputs, compute=lambda: {"schema": "bsbench-local-final-health-v1", "fake": True, "records": fake_records}),
            "final_aware": _local(root, stage="final-aware", model=model, data=data, method=method, prompts=plan_prompts, config=final_inputs, compute=lambda: {"schema": "bsbench-local-final-aware-v1", "fake": True, "records": fake_records}),
            "final_blind": _local(root, stage="final-blind", model=model, data=data, method=method, prompts=plan_prompts, config=final_inputs, compute=lambda: {"schema": "bsbench-local-final-blind-v1", "fake": True, "records": fake_records})}
