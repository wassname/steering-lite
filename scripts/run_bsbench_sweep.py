"""Write the non-paying BS-bench preflight or import a completed production stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from http.client import RemoteDisconnected
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from steering_lite.benchmark.adapters import openrouter_request_callback, real_adapters
from steering_lite.benchmark.cache import committed, content_key, save_json, settle_receipt, source_hash, require_resolved_ledger
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import record_completed_stage, run_condition, run_stages
from steering_lite.benchmark.sweep import JUDGE_MODEL, MODEL_ID, RANDOM_SEEDS, dry_manifest, load_judge_pricing, validate_methods


def run_full_sweep(root: Path, ledger: Path, *, model: dict, rows: list[dict], backend, prompt_spec: dict, judge, methods: tuple[str, ...] = METHODS, measure=None, solver=None, vector_loader=None, transfer_records=None) -> dict:
    """Run named conditions in order and atomically save only a matching run summary."""
    methods = validate_methods(methods)
    summary_path = root / "run-summary.json"
    identity = {
        "schema": "bsbench-run-summary-identity-v1",
        "model": model,
        "cohort": cohort_identity(rows),
        "prompt_spec": prompt_spec,
        "judge": {"model": model["judge_model"], "endpoint": judge.endpoint},
        "methods": list(methods),
        "code_sha256": content_key({
            "benchmark": source_hash(),
            "entrypoint": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }),
    }
    prior = json.loads(summary_path.read_text()) if summary_path.exists() else None
    conditions = dict(prior["conditions"]) if prior and prior.get("identity") == identity else {}
    for method in methods:
        replicates = []
        for random_seed in (RANDOM_SEEDS if method == "random" else (0,)):
            result = run_condition(
            root,
            ledger,
            model=model,
            data=cohort_identity(rows),
            method=method,
            rows=rows,
            backend=backend,
            prompt_spec=prompt_spec,
            judge=judge,
            measure=measure,
            solver=solver,
            vector_loader=vector_loader,
            transfer_records=transfer_records,
            random_seed=random_seed,
            )
            replicates.append(result)
            conditions[method] = {"paid_execution_enabled": result["paid_execution_enabled"], "replicates": list(replicates)} if method == "random" else result
            save_json(summary_path, {"schema": "bsbench-run-summary-v1", "identity": identity, "identity_sha256": content_key(identity), "methods": list(methods), "conditions": conditions})
        if method == "random" and len({replicate["candidate"]["vector_sha256"] for replicate in replicates}) != len(RANDOM_SEEDS):
            raise ValueError("random seeds must produce distinct vector hashes")
        save_json(summary_path, {"schema": "bsbench-run-summary-v1", "identity": identity, "identity_sha256": content_key(identity), "methods": list(methods), "conditions": conditions})
    return {"schema": "bsbench-run-summary-v1", "identity": identity, "identity_sha256": content_key(identity), "methods": list(methods), "conditions": conditions, "summary_path": str(summary_path), "paid_execution_enabled": bool(getattr(backend, "paid_execution_enabled", False))}


def _redact(value):
    if isinstance(value, dict):
        return {
            key: "<redacted>" if key.lower() in {"api_key", "key", "token", "access_token", "refresh_token", "secret", "authorization", "user_id"} else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _provider_headers(headers) -> dict:
    return {
        key.lower(): value
        for key, value in headers.items()
        if key.lower() in {"date", "content-type", "content-length", "retry-after", "x-request-id", "cf-ray"}
        or key.lower().startswith("x-ratelimit-")
    }


def _request_identity(payload: dict) -> dict:
    return {
        "payload_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "model": payload["model"],
        "response_schema": payload["response_format"]["json_schema"]["name"],
    }


@contextmanager
def _openrouter_read_timeout(seconds: float):
    """Apply the fixed read timeout without changing production-cache source identity."""
    if seconds <= 0:
        raise ValueError("OpenRouter read timeout must be positive")
    import steering_lite.benchmark.adapters as adapters

    original_urlopen = adapters.urlopen

    def timed_urlopen(request, *args, **kwargs):
        return original_urlopen(request, *args, **(kwargs | {"timeout": seconds}))

    adapters.urlopen = timed_urlopen
    try:
        yield
    finally:
        adapters.urlopen = original_urlopen


def audited_openrouter_request_callback(*, endpoint: str, api_key: str, evidence_root: Path):
    request_call = openrouter_request_callback(endpoint=endpoint, api_key=api_key)
    lock = threading.Lock()
    sequence = time.time_ns()

    def call(payload: dict) -> dict:
        nonlocal sequence
        with lock:
            sequence += 1
            attempt = sequence
        dispatch_started_monotonic = time.monotonic()
        dispatch_started_at = datetime.now(timezone.utc).isoformat()
        identity = _request_identity(payload)

        def timing(*, schema: str, outcome: str) -> dict:
            response_finished_monotonic = time.monotonic()
            response_finished_at = datetime.now(timezone.utc).isoformat()
            return {
                "schema": schema,
                "recorded_at": response_finished_at,
                "endpoint": endpoint,
                **identity,
                "attempt": attempt,
                "outcome": outcome,
                "dispatch_started_at": dispatch_started_at,
                "response_finished_at": response_finished_at,
                "elapsed_seconds": response_finished_monotonic - dispatch_started_monotonic,
                "enforced_wait_seconds": 0.0,
            }

        def persist(evidence: dict) -> Path:
            path = evidence_root / f"{identity['payload_sha256']}-{attempt:06d}-{evidence['outcome']}.json"
            save_json(path, evidence)
            return path

        try:
            response = request_call(payload)
        except HTTPError as error:
            raw_body = error.read().decode(errors="replace")
            try:
                body = _redact(json.loads(raw_body))
            except json.JSONDecodeError:
                body = {"raw_body": raw_body[:10_000]}
            evidence = timing(schema="bsbench-openrouter-http-error-v1", outcome="failed") | {
                "status": error.code,
                "reason": error.reason,
                "headers": _provider_headers(error.headers),
                "body": body,
            }
            error.evidence_path = str(persist(evidence))
            raise
        except (TimeoutError, URLError, RemoteDisconnected, ConnectionResetError) as error:
            evidence = timing(schema="bsbench-openrouter-no-response-v1", outcome="failed") | {
                "exception_type": type(error).__name__,
            }
            error.evidence_path = str(persist(evidence))
            raise
        except Exception as error:
            evidence = timing(schema="bsbench-openrouter-callback-error-v1", outcome="failed") | {
                "exception_type": type(error).__name__,
            }
            if hasattr(error, "evidence"):
                evidence["parse_evidence"] = _redact(error.evidence)
            error.evidence_path = str(persist(evidence))
            raise
        else:
            persist(timing(schema="bsbench-openrouter-response-timing-v1", outcome="success"))
            return response

    return call


def openrouter_metadata(*, api_key: str) -> dict:
    def get(resource: str) -> dict:
        request = Request(f"https://openrouter.ai/api/v1/{resource}", headers={"Authorization": f"Bearer {api_key}"}, method="GET")
        try:
            with urlopen(request, timeout=30) as response:
                return {"status": response.status, "headers": _provider_headers(response.headers), "body": _redact(json.loads(response.read()))}
        except HTTPError as error:
            raw_body = error.read().decode(errors="replace")
            try:
                body = _redact(json.loads(raw_body))
            except json.JSONDecodeError:
                body = {"raw_body": raw_body[:10_000]}
            return {"status": error.code, "reason": error.reason, "headers": _provider_headers(error.headers), "body": body}

    return {
        "schema": "bsbench-openrouter-nongeneration-metadata-v1",
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": {"key": get("key"), "credits": get("credits"), "models": get("models")},
    }


def import_recorded_smoke(out: Path, model_id: str) -> dict:
    """Import the already-completed Modal smoke; do not dispatch another GPU job."""
    summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
    if summary["model"] != model_id:
        raise ValueError(f"recorded smoke model {summary['model']!r} does not match {model_id!r}")
    rows = read_dev_cohort()[:1]
    if summary["question_id"] != rows[0]["question_id"]:
        raise ValueError("recorded smoke question does not match the fixed development cohort")
    model = {"id": model_id}
    data = cohort_identity(rows)
    prompts = [rows[0]["prompt"]]
    records = []
    for method in summary["conditions"]:
        records.append(record_completed_stage(
            out,
            stage="generation",
            model=model,
            data=data,
            method=method,
            config={"recorded_modal_app": summary["modal_app"], "question_id": summary["question_id"]},
            prompts=prompts,
            result=summary["generation_records"][method] | {"recorded_modal_smoke": summary, "method": method},
        ))
    return {"mode": "run-recorded-smoke", "records": records, "production_cache": str(out / "cache")}


def run_provider_probe(root: Path, ledger: Path, *, model_id: str, endpoint: str, api_key: str, budget: dict, aware_once: bool = False) -> dict:
    rows = read_dev_cohort()
    from steering_lite.benchmark.production import migrate_direct_generation
    from steering_lite.benchmark.cache import peek_stage
    from steering_lite.benchmark.sweep import MODAL_GPU_STAGE_UPPER_USD, PERSONA_VALIDATION_PROMPT_IDS, persona_extraction_identity
    generations = {}
    for method in ("bare", "prompting"):
        config = {"upper_usd": MODAL_GPU_STAGE_UPPER_USD, "prompt_spec": {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128}, "condition": method, "prompt_ids": [row["question_id"] for row in rows]}
        if method == "prompting":
            config |= {"persona_source": persona_extraction_identity(), "persona_validation_prompt_ids": list(PERSONA_VALIDATION_PROMPT_IDS)}
        inputs = {"method": method, "model": {"id": model_id}, "data": cohort_identity(rows), "config": config, "prompts": [row["prompt"] for row in rows]}
        if not migrate_direct_generation(root, **inputs):
            raise ValueError("probe requires validated cached direct generations")
        generation = peek_stage(root / "cache", stage="generation", **inputs)
        if len(generation["answers"]) != len(rows):
            raise ValueError("probe requires complete cached direct generation")
        generations[method] = generation
    from steering_lite.benchmark.validation import numbered_requests, response_record
    row = rows[0] | {"bare": generations["bare"]["answers"][0], "steered": generations["prompting"]["answers"][0], "method": "prompting", "magnitude": None, "random_seed": 0, "side": "+C"}
    requests = numbered_requests([row], JUDGE_MODEL, endpoint)
    if aware_once:
        from steering_lite.benchmark.adapters import judge_request_upper_usd, RunGate
        from steering_lite.benchmark.cache import reserve, settle, estimate_at_reservation_upper, valid_cost
        RunGate(True, budget).require()
        request = next(request for request in requests if not request["blind"])
        reservation = reserve(ledger, "aware-diagnostic-once", judge_request_upper_usd(request), limit_usd=budget["limit_usd"] - budget["external_committed_usd"])
        callback = audited_openrouter_request_callback(endpoint=endpoint, api_key=api_key, evidence_root=root / "provider-evidence")
        evidence = {"schema": "bsbench-aware-diagnostic-once-v1", "request": request, "reservation": reservation}
        try:
            response = callback(request["payload"])
        except Exception as error:
            evidence |= {"success": False, "exception_type": type(error).__name__, "provider_evidence": getattr(error, "evidence_path", None)}
            estimate_at_reservation_upper(ledger, reservation, evidence)
            save_json(root / "aware-diagnostic-once.json", evidence)
            raise
        actual = response.get("_remote_cost_usd")
        if valid_cost(actual):
            settle(ledger, reservation, actual)
        else:
            estimate_at_reservation_upper(ledger, reservation, {"schema": "bsbench-aware-diagnostic-missing-cost-v1", "cost_repr": repr(actual)})
        evidence |= {"success": True, "response": response}
        save_json(root / "aware-diagnostic-once.json", evidence)
        return evidence
    _, judge = real_adapters(modal_stage_call=lambda **_: (_ for _ in ()).throw(RuntimeError("probe must not dispatch Modal")), judge_request_call=audited_openrouter_request_callback(endpoint=endpoint, api_key=api_key, evidence_root=root / "provider-evidence"), judge_endpoint=endpoint, explicit_run=True, budget_preflight=budget, root=root, ledger=ledger)
    before = committed(ledger)
    responses = judge.complete(requests)
    require_resolved_ledger(ledger)
    result = {"schema": "bsbench-settled-provider-probe-v1", "model": JUDGE_MODEL, "question_id": row["question_id"], "request_count": len(requests), "cost_committed_usd": committed(ledger) - before, "requests": requests, "responses": [response_record(request, response) for request, response in zip(requests, responses, strict=True)]}
    save_json(root / "provider-probe.json", result)
    return {key: value for key, value in result.items() if key not in {"requests", "responses"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--import-smoke", action="store_true")
    mode.add_argument("--import-receipt", type=Path)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--probe", action="store_true")
    mode.add_argument("--probe-aware-once", action="store_true")
    mode.add_argument("--check-openrouter-env", action="store_true")
    mode.add_argument("--openrouter-metadata", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("outputs/bsbench-v2"))
    parser.add_argument("--method", "--stage", dest="method", choices=METHODS, help="Run one condition for recovery or debugging; omit for the canonical full sweep.")
    parser.add_argument("--backend", choices=("recorded", "fake", "real"))
    parser.add_argument("--judge-endpoint", default="https://openrouter.ai/api/v1/chat/completions")
    parser.add_argument("--openrouter-read-timeout", type=float, default=180.0)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument("--judge-pricing", type=Path)
    args = parser.parse_args()
    if (args.probe or args.probe_aware_once) and args.method:
        parser.error("--method cannot scope provider probes")
    methods = (args.method,) if args.method else METHODS
    if args.judge_pricing is not None:
        load_judge_pricing(args.judge_pricing)
    if args.dry_run:
        result = dry_manifest(args.out, args.model, ledger=args.ledger, cache_aware=True, judge_endpoint=args.judge_endpoint, methods=methods)
    elif args.probe or args.probe_aware_once:
        api_key = os.environ["OPENROUTER_API_KEY"]
        ledger = args.ledger or args.out / "costs.jsonl"
        budget = dry_manifest(args.out, args.model, ledger=ledger, cache_aware=True, judge_endpoint=args.judge_endpoint, methods=methods)["cost_estimate"]
        if not budget["paid_preflight_passed"]:
            raise RuntimeError("provider probe requires a passing corrected preflight")
        with _openrouter_read_timeout(args.openrouter_read_timeout):
            result = run_provider_probe(args.out, ledger, model_id=args.model, endpoint=args.judge_endpoint, api_key=api_key, budget=budget, aware_once=args.probe_aware_once)
    elif args.check_openrouter_env:
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        result = {"mode": "check-openrouter-env", "openrouter_api_key_present": True}
    elif args.openrouter_metadata:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        evidence = openrouter_metadata(api_key=api_key)
        path = args.metadata_out or args.out / "provider-evidence" / f"openrouter-metadata-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        save_json(path, evidence)
        result = {"mode": "openrouter-metadata", "evidence_path": str(path), "statuses": {name: endpoint["status"] for name, endpoint in evidence["endpoints"].items()}}
    elif args.import_smoke:
        result = import_recorded_smoke(args.out, args.model)
    elif args.import_receipt:
        receipt = json.loads(args.import_receipt.read_text())
        if set(receipt) != {"reservation", "actual_usd", "receipt"}:
            raise ValueError("receipt JSON must contain reservation, actual_usd and receipt")
        ledger = args.ledger or args.out / "costs.jsonl"
        settle_receipt(ledger, receipt["reservation"], receipt["actual_usd"], receipt["receipt"])
        result = {"mode": "receipt-import", "ledger": str(ledger), "reservation": receipt["reservation"], "actual_usd": receipt["actual_usd"]}
    else:
        if args.backend == "fake":
            rows = read_dev_cohort()
            class FakeBackend:
                """Deterministic offline remote-contract backend; its records are not experiment results."""
                remote_vector_binding = True

                def __init__(self): self.calls = []

                def gpu(self, *, stage, method, config, prompts):
                    self.calls.append(stage)
                    if stage == "generation":
                        result = {"actual_usd": 0.0, "answers": [f"Fake {method} answer." for _ in prompts], "health_records": [{"question_id": prompt_id, "reasons": []} for prompt_id in config["prompt_ids"]]}
                        if method == "prompting" and "persona_validation_prompt_ids" in config:
                            result["persona_validation_pairs"] = [
                                {"question_id": prompt_id, "sycophantic": "Fake agreement.", "abrasive": "Fake challenge."}
                                for prompt_id in config["persona_validation_prompt_ids"]
                            ]
                        return result
                    if stage == "calibration-candidates":
                        magnitudes = [0.2, 0.4]
                        spec = config["signed_method_spec"]
                        return {"actual_usd": 0.0, "vector_bytes": f"offline-fake-vector-{method}-{spec['random_seed']}".encode(), "baseline_answers": ["Fake baseline." for _ in prompts], "candidate_magnitudes": magnitudes, "candidate_health": {f"{magnitude}:{side}": {"reasons": []} for magnitude in magnitudes for side in ("+C", "-C")}, "candidate_items": [{"magnitude": magnitude, "side": side, "prompt_index": index, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "response": f"Fake candidate {side}/{magnitude}/{index}."} for magnitude in magnitudes for side in ("+C", "-C") for index, prompt in enumerate(prompts)], "method_config": {"method": method, "layers": spec["layers"], "seed": spec["random_seed"], "target_layer": 29, "skip_first": 16}}
                    if stage == "final-generation":
                        from steering_lite.benchmark.cache import content_key
                        from steering_lite.benchmark.dose_search import BENCHMARK_KL_SPEC, PREDICTION_CASES, final_dose_plan
                        from steering_lite.benchmark.transfer_data import load_evaluation_records, load_transfer_records

                        artifact = config["vector_artifact"]
                        assert __import__("hashlib").sha256(__import__("base64").b64decode(artifact["vector_bytes_b64"])).hexdigest() == artifact["sha256"]
                        target = {"target_id": content_key({"vector": artifact["sha256"], "kl_spec": BENCHMARK_KL_SPEC}), "target_stat": "kl_rms", "target_rms": 1.0, "kl_spec": BENCHMARK_KL_SPEC, "source": {"method": method, "model": args.model}}
                        predictions = [{"schema": "bsbench-signed-rms-kl-transfer-v2", "target_id": target["target_id"], "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)}, "method": method, "model": args.model, "random_seed": config["signed_method_spec"]["random_seed"], "target_stat": "kl_rms", "target_rms": 1.0, "bracket": BENCHMARK_KL_SPEC["bracket"], "kl_spec": BENCHMARK_KL_SPEC, "signed_predictions": [{"side": side, "magnitude": magnitude, "search_history": []} for side, magnitude in (("+C", 0.3), ("-C", 0.35))]} for case in PREDICTION_CASES]
                        plans = [final_dose_plan(prediction) for prediction in predictions]
                        records = {PREDICTION_CASES[0].case_id: load_evaluation_records()} | load_transfer_records(); by_case = {plan["case"]["case_id"]: plan for plan in plans}
                        plan = [{"case_id": case.case_id, "target_id": target["target_id"], "magnitude": dose["magnitude"], "side": dose["side"], "multiplier": dose["multiplier"], "prompt_id": record.prompt_id, "prompt": record.prompt, "prompt_sha256": record.content_sha256} for case in PREDICTION_CASES for record in records[case.case_id] for dose in by_case[case.case_id]["coefficients"]]
                        return {"actual_usd": 0.0, "target": target, "transfer_predictions": predictions, "final_dose_plans": plans, "executable_generation_plan": plan, "baseline_answers": {item["prompt_id"]: "Fake baseline." for item in plan}, "answers": ["Fake final answer." for _ in plan], "health_records": [{"case_id": item["case_id"], "prompt_id": item["prompt_id"], "magnitude": item["magnitude"], "side": item["side"], "reasons": []} for item in plan], "plan_sha256": content_key({"plan": plan})}
                    raise AssertionError(stage)
            class FakeJudge:
                endpoint = "offline-fake-judge"

                def __init__(self): self.calls = 0

                def complete(self, requests):
                    self.calls += len(requests)
                    responses = []
                    for request in requests:
                        if request["schema"] == "bsbench-persona-validation-request-v1":
                            responses.append({"intended_behavior_explains": True, "reason": "The paired responses differ on the stated premise."})
                        elif request["blind"]:
                            responses.append({"summary": "offline difference", "changes": []})
                        else:
                            seed = int(hashlib.sha256(request["comparison_id"].encode()).hexdigest()[:8], 16)
                            effect = .15 + (seed % 70) / 100
                            off_target = .02 + ((seed // 70) % 20) / 100
                            if request["order"] == "AB":
                                responses.append({"evidence": "Fake contrast.", "on_axis_A": 0.0, "on_axis_B": effect, "off_axis_A": 0.0, "off_axis_B": off_target})
                            else:
                                responses.append({"evidence": "Fake contrast.", "on_axis_A": effect, "on_axis_B": 0.0, "off_axis_A": off_target, "off_axis_B": 0.0})
                    return responses

            backend = FakeBackend()
            ledger = args.ledger or args.out / "costs.jsonl"
            common = dict(model={"id": args.model, "judge_model": "offline-fake-judge"}, rows=rows, backend=backend, prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}, judge=FakeJudge(), methods=methods)
            first = run_full_sweep(args.out, ledger, **common)
            calls_after_first = list(backend.calls)
            judge_calls_after_first = common["judge"].calls
            ledger_after_first = ledger.read_bytes()
            second = run_full_sweep(args.out, ledger, **common)
            if backend.calls[len(calls_after_first):] or common["judge"].calls != judge_calls_after_first or ledger.read_bytes() != ledger_after_first:
                raise RuntimeError("immediate fake rerun must make zero GPU/judge calls and no ledger writes")
            from steering_lite.benchmark.production import vector_cached_work
            from steering_lite.benchmark.dose_search import CALIBRATION_CASE
            cached_paid_stages = {}
            for method in methods:
                if method in {"bare", "prompting"}:
                    continue
                for seed in (RANDOM_SEEDS if method == "random" else (0,)):
                    hits, _ = vector_cached_work(args.out, model=common["model"], data=cohort_identity(rows), method=method, random_seed=seed, calibration_prompts=[row["prompt"] for row in rows if row["question_id"] in CALIBRATION_CASE.prompt_ids], prompt_spec=common["prompt_spec"], judge_endpoint=common["judge"].endpoint)
                    if hits != {"calibration-candidates", "candidate-aware", "final-generation", "final-aware", "final-blind"}:
                        raise RuntimeError("completed vector must have no remaining paid stages in preflight")
                    cached_paid_stages[f"{method}/{seed}"] = sorted(hits)
            result = {"mode": "offline-fake-full-sweep", "not_experimental_results": True,
                      "first_backend_calls": calls_after_first, "immediate_rerun_backend_calls": backend.calls[len(calls_after_first):],
                      "first_judge_calls": judge_calls_after_first, "immediate_rerun_judge_calls": common["judge"].calls - judge_calls_after_first,
                      "summary_path": second["summary_path"], "condition_count": len(second["conditions"]), "preflight_cached_paid_stages": cached_paid_stages}
        elif args.backend == "real":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("real backend requires OPENROUTER_API_KEY before remote callbacks are constructed")
            ledger = args.ledger or args.out / "costs.jsonl"
            manifest = dry_manifest(args.out, args.model, ledger=ledger, cache_aware=True, judge_endpoint=args.judge_endpoint, methods=methods)
            budget = manifest["cost_estimate"]
            if not budget["paid_preflight_passed"]:
                raise RuntimeError(f"corrected remaining-work preflight exceeds budget: ${budget['total_upper_usd']:.6f}")
            from run_bsbench_modal import app, remote_stage_call
            with app.run(), _openrouter_read_timeout(args.openrouter_read_timeout):
                modal_adapter, judge_adapter = real_adapters(
                    modal_stage_call=remote_stage_call(args.model, explicit_run=args.run, budget_preflight=budget),
                    judge_request_call=audited_openrouter_request_callback(endpoint=args.judge_endpoint, api_key=api_key, evidence_root=args.out / "provider-evidence"),
                    judge_endpoint=args.judge_endpoint,
                    explicit_run=args.run,
                    budget_preflight=budget,
                    root=args.out,
                    ledger=ledger,
                )
                rows = read_dev_cohort()
                result = run_full_sweep(
                    args.out,
                    ledger,
                    model={"id": args.model, "judge_model": JUDGE_MODEL},
                    rows=rows,
                    backend=modal_adapter,
                    prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128},
                    judge=judge_adapter,
                    methods=methods,
                )
        else:
            summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
            if summary["model"] != args.model:
                raise ValueError("recorded backend model does not match the completed smoke")
            rows = read_dev_cohort()[:1]
            class RecordedBackend:
                def gpu(self, **_kwargs):
                    return summary["generation_records"][args.method or "vjp_cache"] | {"actual_usd": 0.0}
            ledger = args.ledger or args.out / "costs.jsonl"
            smoke_ledger = Path("outputs/bsbench-smoke/costs.jsonl")
            external = committed(smoke_ledger) if ledger != smoke_ledger else 0.0
            if committed(ledger) + external + 0.001 >= 50.0:
                raise RuntimeError("budget includes the unresolved smoke reservation")
            result = run_stages(args.out, ledger, [{"stage": "generation", "model": {"id": args.model}, "data": cohort_identity(rows), "method": args.method or "vjp_cache", "config": {"upper_usd": 0.001, "recorded_modal_app": summary["modal_app"]}, "prompts": [rows[0]["prompt"]]}], RecordedBackend())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
