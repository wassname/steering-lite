import json
from types import SimpleNamespace

import pytest

from steering_lite.benchmark import adapters
from steering_lite.benchmark.adapters import openrouter_request_callback, real_adapters
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.production import _candidate_judgments, run_condition
from steering_lite.benchmark.transfer_data import load_evaluation_records, load_transfer_records


class FakeModalRunMethod:
    def __init__(self):
        self.calls = []

    def gpu(self, *, stage, method, config, prompts):
        self.calls.append((stage, method, tuple(prompts), config))
        if stage == "generation":
            result = {"actual_usd": 0.0, "answers": [f"{method} answer {index}." for index in range(len(prompts))]}
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
            assert "backend_path" not in config["vector_artifact"]
            assert config["vector_artifact"]["vector_bytes_b64"]
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
            if request["schema"] == "bsbench-persona-validation-request-v1":
                responses.append({"intended_behavior_explains": True, "reason": "The paired responses differ on the premise."})
            elif request["blind"]:
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
            tmp_path,
            tmp_path / "ledger.jsonl",
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
        assert result["target"]["observed_boundary"]["generation_health"]["reasons"] == []
        assert result["target"]["observed_boundary"]["generation_health"]["source"] == "fake-health"
        assert len(result["final"]["answers"]) == 84
        assert result["final_health"]["fake"] is False
        assert len(result["final_aware"]["records"]) == len(result["final_blind"]["records"]) == 168
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
            tmp_path,
            tmp_path / "ledger.jsonl",
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


def test_remote_vector_final_binding_uses_portable_bytes_and_no_local_measurement(tmp_path):
    from steering_lite.benchmark.dose_search import PREDICTION_CASES, final_dose_plan

    class RemoteVectorBackend(FakeModalRunMethod):
        remote_vector_binding = True

        def gpu(self, **kwargs):
            if kwargs["stage"] != "final-generation":
                return super().gpu(**kwargs)
            self.calls.append((kwargs["stage"], kwargs["method"], tuple(kwargs["prompts"]), kwargs["config"]))
            config = kwargs["config"]
            artifact = config["vector_artifact"]
            assert "backend_path" not in artifact and artifact["vector_bytes_b64"]
            target = {"target_id": "remote-target", "target_stat": "kl_rms", "target_rms": 1.0}
            predictions = [
                {
                    "schema": "bsbench-rms-kl-transfer-v1",
                    "target_id": target["target_id"],
                    "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)},
                    "method": kwargs["method"],
                    "model": "fake",
                    "target_stat": "kl_rms",
                    "target_rms": 1.0,
                    "bracket": (0.01, 2.0),
                    "predicted_coefficient": 0.3,
                    "search_history": [{"coeff": 0.3, "kl_rms": 1.0}],
                }
                for case in PREDICTION_CASES
            ]
            plans = [final_dose_plan(prediction) for prediction in predictions]
            records = {"bsbench-v2-evaluation": load_evaluation_records()} | load_transfer_records()
            by_case = {plan["case"]["case_id"]: plan for plan in plans}
            plan = [
                {"case_id": case.case_id, "target_id": target["target_id"], "coefficient": coefficient, "prompt_id": record.prompt_id, "prompt": record.prompt, "prompt_sha256": record.content_sha256}
                for case in PREDICTION_CASES for record in records[case.case_id] for coefficient in by_case[case.case_id]["coefficients"]
            ]
            return {
                "cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": 1.0}},
                "target": target,
                "transfer_predictions": predictions,
                "final_dose_plans": plans,
                "executable_generation_plan": plan,
                "baseline_answers": {item["prompt_id"]: "baseline." for item in plan},
                "answers": ["answer." for _ in plan],
                "health_records": [{"case_id": item["case_id"], "prompt_id": item["prompt_id"], "coefficient": item["coefficient"], "reasons": []} for item in plan],
                "plan_sha256": __import__("steering_lite.benchmark.cache", fromlist=["content_key"]).content_key({"plan": plan}),
            }

    rows = read_dev_cohort()
    modal = RemoteVectorBackend()
    result = run_condition(
        tmp_path,
        tmp_path / "ledger.jsonl",
        model={"id": "fake", "judge_model": "fake-judge"},
        data=cohort_identity(rows),
        method="vjp_cache",
        rows=rows,
        backend=modal,
        prompt_spec={"max_new_tokens": 8},
        measure=None,
        solver=None,
        vector_loader=None,
        judge=FakeJudge(),
    )
    assert [stage for stage, *_ in modal.calls] == ["calibration-candidates", "final-generation"]
    assert result["target"]["target_id"] == "remote-target"
    assert len(result["transfer_prediction"]["predictions"]) == 5
    assert len(result["final"]["answers"]) == 84
    events = [json.loads(line)["event"] for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert events == ["reserved", "settled", "reserved", "estimated_at_reservation_upper"]

    class WrongPlan(RemoteVectorBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "final-generation":
                result["executable_generation_plan"] = list(reversed(result["executable_generation_plan"]))
            return result

    with pytest.raises(ValueError, match="executable plan differs"):
        run_condition(
            tmp_path / "mismatch",
            tmp_path / "mismatch" / "ledger.jsonl",
            model={"id": "fake", "judge_model": "fake-judge"},
            data=cohort_identity(rows),
            method="vjp_cache",
            rows=rows,
            backend=WrongPlan(),
            prompt_spec={"max_new_tokens": 8},
            measure=None,
            solver=None,
            vector_loader=None,
            judge=FakeJudge(),
        )
    mismatch_events = [json.loads(line)["event"] for line in (tmp_path / "mismatch" / "ledger.jsonl").read_text().splitlines()]
    assert mismatch_events == ["reserved", "settled", "reserved", "unresolved"]


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
    assert events == ["reserved", "settled", "reserved", "unresolved"]


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
    assert not observation["historical_score_positive"]
    assert not observation["historical_off_axis_within_2_5"]
    assert observation["generation_health"]["reasons"] == []


def test_openrouter_callback_sends_exact_payload_once_and_keeps_usage(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({
                "choices": [{"message": {"content": json.dumps({"summary": "difference", "changes": []})}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.01},
            }).encode()

    monkeypatch.setattr(adapters, "urlopen", lambda request, timeout: calls.append((request, timeout)) or Response())
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "persisted request"}],
        "response_format": {"json_schema": {"schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}, "changes": {"type": "array"}},
            "required": ["summary", "changes"],
            "additionalProperties": False,
        }}},
    }
    result = openrouter_request_callback(endpoint="https://example.invalid/v1/chat/completions", api_key="test-key")(payload)
    assert result == {"summary": "difference", "changes": [], "_remote_usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.01}, "_remote_cost_usd": 0.01}
    assert len(calls) == 1 and calls[0][1] == 90
    assert json.loads(calls[0][0].data) == payload


def _judge_request(request_key: str, *, blind: bool) -> dict:
    return {
        "request_key": request_key,
        "blind": blind,
        "payload": {"model": "fake-judge", "messages": [{"role": "user", "content": request_key}]},
    }


def test_judge_requests_reserve_cache_settle_and_reuse_individually(tmp_path):
    calls = []
    modal, judge = real_adapters(
        modal_stage_call=lambda **_kwargs: pytest.fail("Modal must not run"),
        judge_request_call=lambda payload: calls.append(payload) or {"summary": payload["messages"][0]["content"], "changes": [], "_remote_usage": {"cost": 0.001}, "_remote_cost_usd": 0.001},
        judge_endpoint="https://example.invalid",
        explicit_run=True,
        budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
        root=tmp_path,
        ledger=tmp_path / "ledger.jsonl",
    )
    assert modal
    requests = [_judge_request("aware", blind=False), _judge_request("blind", blind=True)]
    first = judge.complete(requests)
    assert [record["summary"] for record in first] == ["aware", "blind"]
    assert len(calls) == 2
    assert judge.complete(requests) == first
    assert len(calls) == 2
    ledger_rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert [row["event"] for row in ledger_rows] == ["reserved", "settled", "reserved", "settled"]
    assert ledger_rows[0]["upper_usd"] > ledger_rows[2]["upper_usd"]
    cached = list((tmp_path / "cache" / "judge-request").glob("*.json"))
    assert len(cached) == 2
    assert all({"request", "response", "usage", "reservation", "upper_usd"}.issubset(json.loads(path.read_text())["result"]) for path in cached)


def test_judge_partial_failure_reuses_completed_request_after_explicit_receipt(tmp_path):
    from steering_lite.benchmark.cache import settle_receipt

    calls = []
    def fail_second(payload):
        calls.append(payload["messages"][0]["content"])
        if len(calls) == 2:
            raise ConnectionError("request status unknown")
        return {"summary": calls[-1], "changes": [], "_remote_usage": {"cost": 0.001}, "_remote_cost_usd": 0.001}

    _, judge = real_adapters(
        modal_stage_call=lambda **_kwargs: pytest.fail("Modal must not run"),
        judge_request_call=fail_second,
        judge_endpoint="https://example.invalid",
        explicit_run=True,
        budget_preflight={"total_upper_usd": 1.0, "limit_usd": 50.0},
        root=tmp_path,
        ledger=tmp_path / "ledger.jsonl",
    )
    requests = [_judge_request("first", blind=False), _judge_request("second", blind=False)]
    with pytest.raises(ConnectionError, match="unknown"):
        judge.complete(requests)
    assert calls == ["first", "second"]
    rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    unresolved = next(row["reservation"] for row in rows if row["event"] == "unresolved")
    with pytest.raises(RuntimeError, match="unresolved remote work"):
        judge.complete(requests)
    assert calls == ["first", "second"]
    settle_receipt(tmp_path / "ledger.jsonl", unresolved, 0.0, {"outcome": "not_sent"})
    assert [record["summary"] for record in judge.complete(requests)] == ["first", "second"]
    assert calls == ["first", "second", "second"]


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
        root=tmp_path,
        ledger=tmp_path / "gate-ledger.jsonl",
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
        root=tmp_path,
        ledger=tmp_path / "gate-ledger.jsonl",
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
        root=tmp_path,
        ledger=tmp_path / "gate-ledger.jsonl",
    )
    assert allowed_modal.gpu(stage="generation", method="bare", config={}, prompts=[]) == {"stage": "generation"}
    assert allowed_judge.complete([]) == []
