import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from steering_lite.benchmark.adapters import real_adapters
from steering_lite.benchmark.cache import content_key
from steering_lite.benchmark.dose_search import BENCHMARK_KL_SPEC, PREDICTION_CASES, final_dose_plan
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.sweep import persona_extraction_identity
from steering_lite.benchmark.transfer_data import load_evaluation_records, load_transfer_records


@pytest.fixture(autouse=True)
def sourced_test_judge_prices(monkeypatch):
    import steering_lite.benchmark.adapters as adapters_module

    monkeypatch.setattr(adapters_module, "JUDGE_INPUT_USD_PER_MTOKEN", 0.0001)
    monkeypatch.setattr(adapters_module, "JUDGE_OUTPUT_USD_PER_MTOKEN", 0.0001)


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
            magnitudes = [0.2, 0.4]
            return {
                "actual_usd": 0.0,
                "vector_bytes": f"{method}-vector".encode(),
                "baseline_answers": ["baseline." for _ in prompts],
                "candidate_magnitudes": magnitudes,
                "candidate_health": {f"{float(magnitude)}:{side}": {"reasons": []} for magnitude in magnitudes for side in ("+C", "-C")},
                "method_config": {"method": method, "layers": [7, 11, 15, 19, 23], "seed": 0, "target_layer": 29, "skip_first": 16},
                "candidate_items": [
                    {
                        "magnitude": magnitude,
                        "side": side,
                        "prompt_index": index,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response": f"candidate {side} {magnitude}.",
                    }
                    for magnitude in magnitudes
                    for side in ("+C", "-C")
                    for index, prompt in enumerate(prompts)
                ],
            }
        if stage == "final-generation":
            artifact = config["vector_artifact"]
            assert "backend_path" not in artifact
            assert hashlib.sha256(base64.b64decode(artifact["vector_bytes_b64"])).hexdigest() == artifact["sha256"]
            target = {"target_id": "fake-target", "target_stat": "kl_rms", "target_rms": 1.0, "kl_spec": BENCHMARK_KL_SPEC, "source": {"method": method, "model": "fake"}}
            predictions = [
                {
                    "schema": "bsbench-signed-rms-kl-transfer-v2",
                    "target_id": target["target_id"],
                    "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)},
                    "method": method,
                    "model": "fake",
                    "target_stat": "kl_rms",
                    "target_rms": 1.0,
                    "bracket": BENCHMARK_KL_SPEC["bracket"],
                    "kl_spec": BENCHMARK_KL_SPEC,
                    "signed_predictions": [{"side": "+C", "magnitude": 0.3, "search_history": []}, {"side": "-C", "magnitude": 0.35, "search_history": []}],
                }
                for case in PREDICTION_CASES
            ]
            dose_plans = [final_dose_plan(prediction) for prediction in predictions]
            records = config["transfer_prompt_records"]
            assert prompts == [record["prompt"] for case in PREDICTION_CASES for record in records[case.case_id]]
            by_case = {plan["case"]["case_id"]: plan for plan in dose_plans}
            plan = [
                {
                    "case_id": case.case_id,
                    "target_id": target["target_id"],
                    "magnitude": dose["magnitude"],
                    "side": dose["side"],
                    "prompt_id": record["prompt_id"],
                    "prompt": record["prompt"],
                    "prompt_sha256": record["content_sha256"],
                }
                for case in PREDICTION_CASES
                for record in records[case.case_id]
                for dose in by_case[case.case_id]["coefficients"]
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
                    {"case_id": item["case_id"], "prompt_id": item["prompt_id"], "magnitude": item["magnitude"], "side": item["side"], "reasons": []}
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
            content = payload["messages"][0]["content"]
            steered_marker = next(marker for marker in ("candidate", "final.", "prompting answer.") if marker in content)
            response_a = content.split("Response A:\n", 1)[1].split("\n\nResponse B:\n", 1)[0]
            if steered_marker in response_a:
                response = {"evidence": "A is the steered response.", "on_axis_A": 1.0, "on_axis_B": 0.0, "off_axis_A": 0.0, "off_axis_B": 0.0}
            else:
                response = {"evidence": "B is the steered response.", "on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.0, "off_axis_B": 0.0}
        return response | {"_remote_usage": {"cost": 0.0}, "_remote_cost_usd": 0.0}


class RandomNamedStageCall(FakeRemoteStageCall):
    """Make the fake random candidate identifiable from its answer text."""

    def __call__(self, *, stage, method, config, prompts):
        result = super().__call__(stage=stage, method=method, config=config, prompts=prompts)
        if stage == "calibration-candidates" and method == "random":
            for item in result["candidate_items"]:
                item["response"] = f"zzrandommarker {item['response']}"
        return result


class NonUsefulRandomJudgeCall(FakeJudgeCall):
    """Score only the synthetic random candidate as valid but non-useful."""

    def __call__(self, payload):
        if payload["response_format"]["json_schema"]["name"] == "demo_rating" and "zzrandommarker" in json.dumps(payload):
            self.payloads.append(payload)
            response = {"evidence": "No measured directed change.", "on_axis_A": 0.0, "on_axis_B": 0.0, "off_axis_A": 0.0, "off_axis_B": 0.0}
            return response | {"_remote_usage": {"cost": 0.0}, "_remote_cost_usd": 0.0}
        return super().__call__(payload)


def _run(root: Path, ledger: Path, stage_call: FakeRemoteStageCall, judge_call: FakeJudgeCall, *, endpoint: str, prompt_spec: dict, transfer_records=None, methods=METHODS):
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
        methods=methods,
    )


def test_negative_candidate_score_does_not_prevent_final_evaluation(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    stage_call = RandomNamedStageCall()
    judge_call = NonUsefulRandomJudgeCall()
    prompt_spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}

    first = _run(
        root, ledger, stage_call, judge_call,
        endpoint="https://judge.example/v1", prompt_spec=prompt_spec,
        methods=("random", "mean_diff"),
    )

    assert stage_call.calls == [
        ("calibration-candidates", "random"),
        ("final-generation", "random"),
        ("calibration-candidates", "mean_diff"),
        ("final-generation", "mean_diff"),
    ]
    assert first["conditions"]["random"]["candidate_judgments"]["observed"]
    assert first["conditions"]["random"]["final"]["reused"] is False
    assert first["conditions"]["mean_diff"]["final"]["reused"] is False

    second = _run(
        root, ledger, stage_call, judge_call,
        endpoint="https://judge.example/v1", prompt_spec=prompt_spec,
        methods=("random", "mean_diff"),
    )
    assert second["conditions"]["random"]["final"]["reused"] is True
    assert len(stage_call.calls) == 4


def test_judge_endpoint_change_reuses_gpu_and_preserves_vector_sidecar(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    stage_call, judge_call = FakeRemoteStageCall(), FakeJudgeCall()
    prompt_spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}
    first = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-a.example/v1", prompt_spec=prompt_spec, methods=("pca",))
    artifact = first["conditions"]["pca"]["candidate"]["vector_artifact"]
    sidecar = root / artifact["path"]
    assert sidecar.is_file() and hashlib.sha256(sidecar.read_bytes()).hexdigest() == artifact["sha256"]
    gpu_calls = list(stage_call.calls)
    judge_calls = len(judge_call.payloads)
    cache_counts = {path.name: len(list(path.glob("*.json"))) for path in (root / "cache").iterdir()}

    second = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-b.example/v1", prompt_spec=prompt_spec, methods=("pca",))

    assert stage_call.calls == gpu_calls
    assert len(judge_call.payloads) == 2 * judge_calls
    assert second["conditions"]["pca"]["candidate"]["reused"] is True
    assert second["conditions"]["pca"]["final"]["reused"] is True
    assert sidecar.is_file() and hashlib.sha256(sidecar.read_bytes()).hexdigest() == artifact["sha256"]
    changed_stages = {
        path.name
        for path in (root / "cache").iterdir()
        if len(list(path.glob("*.json"))) != cache_counts.get(path.name, 0)
    }
    assert changed_stages == {"judge-request", "candidate-judgments", "final-judgments", "final-health", "final-aware", "final-blind"}


def test_transfer_prompt_change_invalidates_only_final_gpu_stage(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    stage_call, judge_call = FakeRemoteStageCall(), FakeJudgeCall()
    prompt_spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}
    _run(root, ledger, stage_call, judge_call, endpoint="https://judge.example/v1", prompt_spec=prompt_spec, methods=("pca",))
    changed_records = {case_id: tuple(records) for case_id, records in load_transfer_records().items()}
    case_id = next(iter(changed_records))
    record = changed_records[case_id][0]
    changed_prompt = record.prompt + " Changed transfer prompt."
    changed_records[case_id] = (record.__class__(record.prompt_id, changed_prompt, record.dataset, record.source_path, record.source_revision, record.source_sha256, hashlib.sha256(changed_prompt.encode()).hexdigest(), record.answer_key, record.answer_key_sha256), *changed_records[case_id][1:])
    before = len(stage_call.calls)

    result = _run(root, ledger, stage_call, judge_call, endpoint="https://judge.example/v1", prompt_spec=prompt_spec, transfer_records=changed_records, methods=("pca",))

    assert stage_call.calls[before:] == [("final-generation", "pca")]
    assert result["conditions"]["pca"]["candidate"]["reused"] is True
    assert result["conditions"]["pca"]["final"]["reused"] is False


def test_prompt_spec_change_invalidates_candidate_and_final_gpu_stages(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    stage_call, judge_call = FakeRemoteStageCall(), FakeJudgeCall()
    prompt_spec = {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8}
    _run(root, ledger, stage_call, judge_call, endpoint="https://judge.example/v1", prompt_spec=prompt_spec, methods=("pca",))
    before = len(stage_call.calls)

    result = _run(root, ledger, stage_call, judge_call, endpoint="https://judge.example/v1", prompt_spec=prompt_spec | {"template": "Answer plainly."}, methods=("pca",))

    assert stage_call.calls[before:] == [("calibration-candidates", "pca"), ("final-generation", "pca")]
    assert result["conditions"]["pca"]["candidate"]["reused"] is False
    assert result["conditions"]["pca"]["final"]["reused"] is False


def test_signed_run_persists_both_final_sides_after_negative_calibration(tmp_path: Path):
    root, ledger = tmp_path / "run", tmp_path / "ledger.jsonl"
    summary = _run(
        root,
        ledger,
        RandomNamedStageCall(),
        NonUsefulRandomJudgeCall(),
        endpoint="https://judge.example/v1",
        prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 8},
    )

    random = summary["conditions"]["random"]
    assert random["final"]["reused"] is False
    assert {record["side"] for record in random["final_health"]["records"]} == {"+C", "-C"}
    assert len(random["final_health"]["records"]) == 6 * sum(len(case.prompt_ids) for case in PREDICTION_CASES)


def test_full_entrypoint_runs_canonical_remote_contract_and_reuses(tmp_path: Path):
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
    assert len(judge_call.payloads) == 6756
    validation = first["conditions"]["prompting"]["persona_validation"]
    assert len(validation["comparisons"]) == len(validation["requests"]) == len(validation["results"]) == 12
    assert validation["disagreements"] == []
    assert all(comparison["persona_source"] == persona_extraction_identity() for comparison in validation["comparisons"])
    assert all(request["input_tokens_upper"] == 2_000 and request["output_tokens_upper"] == 100 for request in validation["requests"])
    assert all(response["response"]["intended_behavior_explains"] for response in validation["results"])
    assert all("method" not in payload and "coefficient" not in payload for payload in judge_call.payloads if payload["response_format"]["json_schema"]["name"] == "blind_change_description")

    assert all({record["side"] for record in condition["final_health"]["records"]} == {"+C", "-C"} for condition in first["conditions"].values() if "final_health" in condition)

    first_stage_calls = len(stage_call.calls)
    first_judge_calls = len(judge_call.payloads)
    second = _run(root, ledger, stage_call, judge_call, endpoint="https://judge-a.example/v1", prompt_spec=prompt_spec)
    assert second["conditions"]["bare"]["generation"]["reused"] is True
    assert len(stage_call.calls) == first_stage_calls
    assert len(judge_call.payloads) == first_judge_calls

    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    judge_reservations = [row for row in events if row["event"] == "reserved" and row["kind"].startswith("judge-")]
    judge_settlements = [row for row in events if row["event"] == "settled" and row["reservation"] in {row["id"] for row in judge_reservations}]
    assert len(judge_reservations) == len(judge_settlements)
    assert all(row["actual_usd"] == 0.0 for row in judge_settlements)
    assert (root / "run-summary.json").exists()
