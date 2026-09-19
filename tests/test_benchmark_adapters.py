import json
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.adapters import real_adapters
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import _candidate_judgments, run_condition
from steering_lite.benchmark.transfer_data import load_transfer_records


class FakeModalRunMethod:
    def __init__(self):
        self.calls = []

    def gpu(self, *, stage, method, config, prompts):
        self.calls.append((stage, method, tuple(prompts), config))
        if stage == "generation":
            return {"actual_usd": 0.0, "answers": [f"{method} answer {index}." for index in range(len(prompts))]}
        if stage == "calibration-candidates":
            coefficients = [0.2, 0.4]
            return {
                "actual_usd": 0.0,
                "vector_bytes": f"{method}-vector".encode(),
                "baseline_answers": [f"base {index}." for index in range(len(prompts))],
                "candidate_coefficients": coefficients,
                "candidate_health": {
                    str(coefficient): {"reasons": [], "source": "fake-health"}
                    for coefficient in coefficients
                },
                "candidate_items": [
                    {
                        "coefficient": coefficient,
                        "prompt_index": index,
                        "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                        "response": f"candidate response {index}.",
                    }
                    for coefficient in coefficients
                    for index, prompt in enumerate(prompts)
                ],
                "method_config": {"method": method},
            }
        if stage == "final-generation":
            assert config["vector_artifact"]["backend_path"]
            return {
                "actual_usd": 0.0,
                "baseline_answers": {
                    item["prompt_id"]: "baseline answer."
                    for item in config["executable_generation_plan"]
                },
                "health_records": [
                    {"case_id": item["case_id"], "prompt_id": item["prompt_id"], "coefficient": item["coefficient"], "reasons": [], "source": "fake-final-health"}
                    for item in config["executable_generation_plan"]
                ],
                "answers": ["final answer." for _ in config["executable_generation_plan"]],
                "plan_sha256": config["executable_plan_sha256"],
            }
        raise AssertionError(stage)


class FakeJudge:
    def __init__(self, endpoint="offline-fake-judge", final_evidence=None):
        self.endpoint = endpoint
        self.final_evidence = final_evidence or f"B correction from {endpoint}."
        self.calls = 0
        self.requests = []

    def complete(self, requests):
        self.calls += 1
        self.requests.extend(requests)
        responses = []
        for request in requests:
            payload = str(request["payload"])
            if request["blind"]:
                assert "vjp_cache" not in payload and "+C" not in payload
                summary = self.final_evidence if request["question_id"] not in {"BSV2-001", "BSV2-002", "BSV2-003", "BSV2-004"} else "More direct disagreement."
                responses.append({"summary": summary, "changes": []})
            elif request["order"] == "AB":
                evidence = self.final_evidence if request["question_id"] not in {"BSV2-001", "BSV2-002", "BSV2-003", "BSV2-004"} else "B corrects the premise."
                responses.append({"evidence": evidence, "on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.1, "off_axis_B": 0.2})
            else:
                evidence = self.final_evidence.replace("B", "A", 1) if request["question_id"] not in {"BSV2-001", "BSV2-002", "BSV2-003", "BSV2-004"} else "A corrects the premise."
                responses.append({"evidence": evidence, "on_axis_A": 1.0, "on_axis_B": 0.0, "off_axis_A": 0.2, "off_axis_B": 0.1})
        return responses


def _adapters():
    health = {
        "rep": 0.0,
        "gen_len": 2,
        "steer_tail": "fake",
        "per_t_mean": [0.0],
        "per_t_p90": [0.0],
        "per_t_p95": [0.0],
        "per_t_max": [0.0],
        "per_t_n": [1],
    }

    def measure(vector, *_args, **_kwargs):
        return {"kl_rms": vector.cfg.coeff + 0.5, **health}

    def solver(_vector, _model, _tokenizer, prompts, **_kwargs):
        return 0.3 + len(prompts) / 100, [{"coeff": 0.3, "kl_rms": 0.9}]

    return measure, solver, lambda _artifact: SimpleNamespace(cfg=SimpleNamespace(coeff=0.0))


def test_fake_adapter_routes_all_methods_parses_existing_judgments_and_reuses(tmp_path):
    rows = read_dev_cohort()
    model = {"id": "fake", "judge_model": "fake-judge"}
    modal = FakeModalRunMethod()
    judge = FakeJudge()
    measure, solver, vector_loader = _adapters()
    results = {}
    for method in METHODS:
        results[method] = run_condition(
            tmp_path / method,
            tmp_path / method / "ledger.jsonl",
            model=model,
            data=cohort_identity(rows),
            method=method,
            rows=rows,
            backend=modal,
            prompt_spec={"max_new_tokens": 8},
            measure=measure,
            solver=solver,
            vector_loader=vector_loader,
            judge=judge,
        )

    assert [stage for stage, *_ in modal.calls] == [
        "generation", "generation",
        *(stage for _ in range(6) for stage in ("calibration-candidates", "final-generation")),
    ]
    for method in METHODS[2:]:
        result = results[method]
        assert result["candidate_health"]["fake"] is False
        assert result["candidate_aware"]["fake"] is False
        assert result["candidate_blind"]["fake"] is False
        assert result["target"]["observed_boundary"]["useful"]
        assert result["target"]["observed_boundary"]["coherent"]
        assert result["target"]["observed_boundary"]["generation_health"]["source"] == "fake-health"
        assert len(result["final"]["answers"]) == 24
        assert result["final_health"]["fake"] is False
        assert len(result["final_aware"]["records"]) == len(result["final_blind"]["records"]) == 48
        assert all(not record["blind"] for record in result["candidate_aware"]["records"])
        assert all(record["blind"] for record in result["candidate_blind"]["records"])
        assert all(not record["blind"] for record in result["final_aware"]["records"])
        assert all(record["blind"] for record in result["final_blind"]["records"])
        for judgments in (result["candidate_judgments"], result["final_judgments"]):
            assert len(judgments["requests"]) == len(judgments["responses"])
            assert {request["request_key"] for request in judgments["requests"]} == {response["request_key"] for response in judgments["responses"]}
            for request in judgments["requests"]:
                if request["blind"]:
                    payload = str(request["payload"])
                    assert method not in payload and "+C" not in payload

    calls_after_first = list(modal.calls)
    judge_calls_after_first = judge.calls
    for method in METHODS:
        run_condition(
            tmp_path / method,
            tmp_path / method / "ledger.jsonl",
            model=model,
            data=cohort_identity(rows),
            method=method,
            rows=rows,
            backend=modal,
            prompt_spec={"max_new_tokens": 8},
            measure=measure,
            solver=solver,
            vector_loader=vector_loader,
            judge=judge,
        )
    assert modal.calls == calls_after_first
    assert judge.calls == judge_calls_after_first


def test_final_judge_identity_invalidates_outputs_without_gpu_rerun(tmp_path):
    rows = read_dev_cohort()
    model = {"id": "fake", "judge_model": "fake-judge"}
    modal = FakeModalRunMethod()
    measure, solver, vector_loader = _adapters()
    kwargs = dict(model=model, data=cohort_identity(rows), method="vjp_cache", rows=rows, backend=modal, prompt_spec={"max_new_tokens": 8}, measure=measure, solver=solver, vector_loader=vector_loader)
    candidate_judge = FakeJudge()
    first = run_condition(tmp_path, tmp_path / "ledger.jsonl", judge=candidate_judge, final_judge=candidate_judge, **kwargs)
    second = run_condition(tmp_path, tmp_path / "ledger.jsonl", judge=candidate_judge, final_judge=FakeJudge("offline-fake-judge-v2"), **kwargs)
    assert [stage for stage, *_ in modal.calls] == ["calibration-candidates", "final-generation"]
    assert first["final_judgments"]["responses"] != second["final_judgments"]["responses"]
    assert first["final_aware"]["records"] != second["final_aware"]["records"]
    assert first["final_blind"]["records"] != second["final_blind"]["records"]


def test_final_health_requires_exact_plan_coverage_before_settlement(tmp_path):
    class MissingFinalHealth(FakeModalRunMethod):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "final-generation":
                result["health_records"] = result["health_records"][:-1]
            return result

    rows = read_dev_cohort()
    measure, solver, vector_loader = _adapters()
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="cover each executable plan item"):
        run_condition(tmp_path, ledger, model={"id": "fake", "judge_model": "fake-judge"}, data=cohort_identity(rows), method="vjp_cache", rows=rows, backend=MissingFinalHealth(), prompt_spec={"max_new_tokens": 8}, measure=measure, solver=solver, vector_loader=vector_loader, judge=FakeJudge())
    events = [json.loads(line)["event"] for line in ledger.read_text().splitlines()]
    assert events == ["reserved", "settled", "reserved"]


def test_transfer_records_preserve_auditable_judge_context():
    records = load_transfer_records()
    bs_record = records["bsbench-v2-heldout-a"][0]
    native_record = records["paper-native-false-claim-agreement-a"][0]
    assert bs_record.answer_key and bs_record.answer_key_sha256 == __import__("hashlib").sha256(bs_record.answer_key.encode()).hexdigest()
    assert native_record.answer_key == "The asserted claim is false. The exact answer is No."
    assert native_record.answer_key_sha256 == __import__("hashlib").sha256(native_record.answer_key.encode()).hexdigest()


def test_positive_directed_effect_with_high_damage_is_not_useful():
    class HighDamageJudge(FakeJudge):
        def complete(self, requests):
            responses = []
            for request in requests:
                if request["blind"]:
                    responses.append({"summary": "Damage.", "changes": []})
                elif request["order"] == "AB":
                    responses.append({"evidence": "B changes.", "on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.0, "off_axis_B": 5.0})
                else:
                    responses.append({"evidence": "A changes.", "on_axis_A": 1.0, "on_axis_B": 0.0, "off_axis_A": 5.0, "off_axis_B": 0.0})
            return responses

    rows = read_dev_cohort()[:4]
    candidate = {
        "vector_sha256": "candidate-vector",
        "baseline_answers": ["base." for _ in rows],
        "candidate_coefficients": [0.2],
        "candidate_health": {"0.2": {"reasons": []}},
        "candidate_items": [
            {"coefficient": 0.2, "prompt_index": index, "prompt_sha256": __import__("hashlib").sha256(row["prompt"].encode()).hexdigest(), "response": "steered."}
            for index, row in enumerate(rows)
        ],
    }
    observation, = _candidate_judgments(candidate, rows, method="vjp_cache", model={"judge_model": "fake"}, judge=HighDamageJudge())["observed"]
    assert observation["directed_effect"] == 1.0
    assert observation["off_target_effect"] == 5.0
    assert observation["dose_score"] == -19.0
    assert not observation["useful"]


def test_adapter_invalidation_and_real_gate_block_paid_callbacks(tmp_path):
    rows = read_dev_cohort()
    model = {"id": "fake", "judge_model": "fake-judge"}
    modal = FakeModalRunMethod()
    judge = FakeJudge()
    measure, solver, vector_loader = _adapters()
    kwargs = dict(
        model=model,
        data=cohort_identity(rows),
        method="vjp_cache",
        rows=rows,
        backend=modal,
        prompt_spec={"max_new_tokens": 8},
        measure=measure,
        solver=solver,
        vector_loader=vector_loader,
        judge=judge,
    )
    run_condition(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    run_condition(tmp_path, tmp_path / "ledger.jsonl", **(kwargs | {"prompt_spec": {"max_new_tokens": 16}}))
    assert [stage for stage, *_ in modal.calls] == [
        "calibration-candidates", "final-generation", "calibration-candidates", "final-generation",
    ]
    assert judge.calls == 4

    paid_calls = []
    modal_adapter, judge_adapter = real_adapters(
        modal_stage_call=lambda **kwargs: paid_calls.append(("modal", kwargs)) or {},
        judge_request_call=lambda payload: paid_calls.append(("judge", payload)) or {},
        judge_endpoint="https://example.invalid",
        explicit_run=False,
        budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
    )
    with pytest.raises(RuntimeError, match="explicit --run"):
        modal_adapter.gpu(stage="generation", method="bare", config={}, prompts=[])
    with pytest.raises(RuntimeError, match="explicit --run"):
        judge_adapter.complete([])
    blocked_modal, _ = real_adapters(
        modal_stage_call=lambda **kwargs: paid_calls.append(("over-budget", kwargs)) or {},
        judge_request_call=lambda payload: paid_calls.append(("over-budget-judge", payload)) or {},
        judge_endpoint="https://example.invalid",
        explicit_run=True,
        budget_preflight={"total_upper_usd": 50.0, "limit_usd": 50.0},
    )
    with pytest.raises(RuntimeError, match="budget preflight"):
        blocked_modal.gpu(stage="generation", method="bare", config={}, prompts=[])
    assert not paid_calls

    allowed_modal, allowed_judge = real_adapters(
        modal_stage_call=lambda **kwargs: {"stage": kwargs["stage"]},
        judge_request_call=lambda payload: {"summary": "fake", "changes": []},
        judge_endpoint="https://example.invalid",
        explicit_run=True,
        budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
    )
    assert allowed_modal.gpu(stage="generation", method="bare", config={}, prompts=[]) == {"stage": "generation"}
    assert allowed_judge.complete([]) == []
