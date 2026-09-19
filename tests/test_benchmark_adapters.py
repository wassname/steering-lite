from types import SimpleNamespace

import pytest

from steering_lite.benchmark.adapters import real_adapters
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import run_condition


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
                "answers": ["final answer." for _ in config["executable_generation_plan"]],
                "plan_sha256": config["executable_plan_sha256"],
            }
        raise AssertionError(stage)


class FakeJudge:
    endpoint = "offline-fake-judge"

    def __init__(self):
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
                responses.append({"summary": "More direct disagreement.", "changes": []})
            elif request["order"] == "AB":
                responses.append({"evidence": "B corrects the premise.", "on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.1, "off_axis_B": 0.2})
            else:
                responses.append({"evidence": "A corrects the premise.", "on_axis_A": 1.0, "on_axis_B": 0.0, "off_axis_A": 0.2, "off_axis_B": 0.1})
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
        assert len(result["final"]["answers"]) == len(result["final_aware"]["records"]) == 24
        assert all(not record["blind"] for record in result["candidate_aware"]["records"])
        assert all(record["blind"] for record in result["candidate_blind"]["records"])

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
