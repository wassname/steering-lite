import base64
import hashlib
import importlib.util
import json
from pathlib import Path

from steering_lite.benchmark.adapters import real_adapters
from steering_lite.benchmark.cache import content_key
from steering_lite.benchmark.dose_search import TRANSFER_CASES, final_dose_plan
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.sweep import persona_extraction_identity
from steering_lite.benchmark.transfer_data import load_transfer_records


def _entrypoint_module():
    path = Path(__file__).parents[1] / "scripts" / "run_bsbench_sweep.py"
    spec = importlib.util.spec_from_file_location("run_bsbench_sweep", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeRemoteStageCall:
    """A zero-cost contract implementation; it never contacts Modal."""

    def __init__(self):
        self.calls = []

    def __call__(self, *, stage, method, config, prompts):
        self.calls.append((stage, method))
        if stage == "generation":
            result = {
                "actual_usd": 0.0,
                "answers": [f"{method} answer." for _ in prompts],
                "health_records": [{"question_id": prompt_id, "reasons": []} for prompt_id in config["prompt_ids"]],
            }
            if method == "prompting" and "persona_validation_prompt_ids" in config:
                result["persona_validation_pairs"] = [
                    {"question_id": prompt_id, "sycophantic": "agreement.", "abrasive": "challenge."}
                    for prompt_id in config["persona_validation_prompt_ids"]
                ]
            return result
        if stage == "calibration-candidates":
            coefficients = [0.2, 0.4]
            return {
                "actual_usd": 0.0,
                "vector_bytes": f"{method}-vector".encode(),
                "baseline_answers": ["baseline." for _ in prompts],
                "candidate_coefficients": coefficients,
                "candidate_health": {str(coefficient): {"reasons": []} for coefficient in coefficients},
                "candidate_items": [
                    {
                        "coefficient": coefficient,
                        "prompt_index": index,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response": f"candidate {coefficient}.",
                    }
                    for coefficient in coefficients
                    for index, prompt in enumerate(prompts)
                ],
            }
        if stage == "final-generation":
            artifact = config["vector_artifact"]
            assert "backend_path" not in artifact
            assert hashlib.sha256(base64.b64decode(artifact["vector_bytes_b64"])).hexdigest() == artifact["sha256"]
            target = {"target_id": "fake-target", "target_stat": "kl_rms", "target_rms": 1.0}
            predictions = [
                {
                    "schema": "bsbench-rms-kl-transfer-v1",
                    "target_id": target["target_id"],
                    "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)},
                    "method": method,
                    "model": "fake",
                    "target_stat": "kl_rms",
                    "target_rms": 1.0,
                    "bracket": (0.01, 2.0),
                    "predicted_coefficient": 0.3,
                    "search_history": [],
                }
                for case in TRANSFER_CASES
            ]
            dose_plans = [final_dose_plan(prediction) for prediction in predictions]
            records = config["transfer_prompt_records"]
            assert prompts == [record["prompt"] for case in TRANSFER_CASES for record in records[case.case_id]]
            by_case = {plan["case"]["case_id"]: plan for plan in dose_plans}
            plan = [
                {
                    "case_id": case.case_id,
                    "target_id": target["target_id"],
                    "coefficient": coefficient,
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "prompt_sha256": record["content_sha256"],
                }
                for case in TRANSFER_CASES
                for record in records[case.case_id]
                for coefficient in by_case[case.case_id]["coefficients"]
            ]
            return {
                "actual_usd": 0.0,
                "target": target,
                "transfer_predictions": predictions,
                "final_dose_plans": dose_plans,
                "executable_generation_plan": plan,
                "baseline_answers": {item["prompt_id"]: "baseline." for item in plan},
                "answers": ["final." for _ in plan],
                "health_records": [
                    {"case_id": item["case_id"], "prompt_id": item["prompt_id"], "coefficient": item["coefficient"], "reasons": []}
                    for item in plan
                ],
                "plan_sha256": content_key({"plan": plan}),
            }
        raise AssertionError(stage)


class FakeJudgeCall:
    """A zero-cost strict-schema response callback; it never contacts OpenRouter."""

    def __init__(self):
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(payload)
        name = payload["response_format"]["json_schema"]["name"]
        if name == "persona_pair_validation":
            response = {"intended_behavior_explains": True, "reason": "The paired responses differ on the premise."}
        elif name == "blind_change_description":
            response = {"summary": "No material change.", "changes": []}
        else:
            response = {"evidence": "B agrees more.", "on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.0, "off_axis_B": 0.0}
        return response | {"_remote_usage": {"cost": 0.0}, "_remote_cost_usd": 0.0}


def _run(root: Path, ledger: Path, stage_call: FakeRemoteStageCall, judge_call: FakeJudgeCall, *, endpoint: str, prompt_spec: dict, transfer_records=None):
    modal, judge = real_adapters(
        modal_stage_call=stage_call,
        judge_request_call=judge_call,
        judge_endpoint=endpoint,
        explicit_run=True,
        budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
        root=root,
        ledger=ledger,
    )
    return _entrypoint_module().run_full_sweep(
        root,
        ledger,
        model={"id": "fake", "judge_model": "fake-judge"},
        rows=read_dev_cohort(),
        backend=modal,
        prompt_spec=prompt_spec,
        judge=judge,
        transfer_records=transfer_records,
    )


def test_full_entrypoint_runs_canonical_remote_contract_and_reuses_then_invalidates_downstream(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    stage_call = FakeRemoteStageCall()
    judge_call = FakeJudgeCall()
    prompt_spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}

    first = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-a.example/v1", prompt_spec=prompt_spec)
    assert first["methods"] == list(METHODS)
    assert list(first["conditions"]) == list(METHODS)
    assert first["paid_execution_enabled"] is True
    assert {result["paid_execution_enabled"] for result in first["conditions"].values()} == {True}
    assert [stage for stage, _method in stage_call.calls] == [
        "generation", "generation",
        *(stage for _ in METHODS[2:] for stage in ("calibration-candidates", "final-generation")),
    ]
    assert len(stage_call.calls) == 14
    assert len(judge_call.payloads) == 860
    validation = first["conditions"]["prompting"]["persona_validation"]
    assert len(validation["examples"]) == len(validation["requests"]) == len(validation["responses"]) == len(validation["disagreements"]) == 12
    assert all(example["persona_source"] == persona_extraction_identity() for example in validation["examples"])
    assert all(request["input_tokens_upper"] == 2_000 and request["output_tokens_upper"] == 100 for request in validation["requests"])
    assert all(response["response"]["intended_behavior_explains"] for response in validation["responses"])
    assert all("method" not in payload and "coefficient" not in payload for payload in judge_call.payloads if payload["response_format"]["json_schema"]["name"] == "blind_change_description")

    first_stage_calls = len(stage_call.calls)
    first_judge_calls = len(judge_call.payloads)
    second = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-a.example/v1", prompt_spec=prompt_spec)
    assert second["conditions"]["bare"]["generation"]["reused"] is True
    assert len(stage_call.calls) == first_stage_calls
    assert len(judge_call.payloads) == first_judge_calls

    _run(root, ledger, stage_call, judge_call, endpoint="https://judge-b.example/v1", prompt_spec=prompt_spec)
    assert len(stage_call.calls) == first_stage_calls
    assert len(judge_call.payloads) == first_judge_calls * 2

    changed_records = {
        case_id: tuple(records)
        for case_id, records in load_transfer_records().items()
    }
    first_case = TRANSFER_CASES[0].case_id
    changed = changed_records[first_case][0]
    changed_records[first_case] = (changed.__class__(
        changed.prompt_id,
        changed.prompt + " Changed transfer prompt.",
        changed.dataset,
        changed.source_path,
        changed.source_revision,
        changed.source_sha256,
        hashlib.sha256((changed.prompt + " Changed transfer prompt.").encode()).hexdigest(),
        changed.answer_key,
        changed.answer_key_sha256,
    ),) + changed_records[first_case][1:]
    before_prompt_change = len(stage_call.calls)
    _run(root, ledger, stage_call, judge_call, endpoint="https://judge-b.example/v1", prompt_spec=prompt_spec, transfer_records=changed_records)
    assert stage_call.calls[before_prompt_change:] == [("final-generation", method) for method in METHODS[2:]]

    recovery = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-b.example/v1", prompt_spec={**prompt_spec, "template": "Answer plainly."})
    assert recovery["methods"] == list(METHODS)
    assert set(recovery["conditions"]) == set(METHODS)
    single_method = _entrypoint_module().run_full_sweep(
        root,
        ledger,
        model={"id": "fake", "judge_model": "fake-judge"},
        rows=read_dev_cohort(),
        backend=real_adapters(
            modal_stage_call=stage_call,
            judge_request_call=judge_call,
            judge_endpoint="https://judge-b.example/v1",
            explicit_run=True,
            budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
            root=root,
            ledger=ledger,
        )[0],
        prompt_spec={**prompt_spec, "template": "Answer in one sentence."},
        judge=real_adapters(
            modal_stage_call=stage_call,
            judge_request_call=judge_call,
            judge_endpoint="https://judge-b.example/v1",
            explicit_run=True,
            budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
            root=root,
            ledger=ledger,
        )[1],
        methods=("pca",),
    )
    assert single_method["methods"] == ["pca"]
    assert set(single_method["conditions"]) == {"pca"}
    assert json.loads((root / "run-summary.json").read_text())["identity"] == single_method["identity"]

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    judge_reservations = [row for row in events if row["event"] == "reserved" and row["kind"].startswith("judge-")]
    judge_settlements = [row for row in events if row["event"] == "settled" and row["reservation"] in {row["id"] for row in judge_reservations}]
    assert len(judge_reservations) == len(judge_settlements)
    assert all(row["actual_usd"] == 0.0 for row in judge_settlements)
    assert (root / "run-summary.json").exists()
