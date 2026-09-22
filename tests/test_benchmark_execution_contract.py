"""Bounded execution regressions for signed BS-bench recovery. — PI/OpenAI"""
import hashlib
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest
import torch

from steering_lite.benchmark import adapters, sweep
from steering_lite.benchmark.adapters import LocalJudgeAdapter, RunGate
from steering_lite.benchmark.cache import committed, reserve, settle
from steering_lite.benchmark.judge import FORMAT, BLIND_FORMAT, score_pair, validate_judgment
from steering_lite.benchmark.validation import comparison_id, numbered_requests
from steering_lite.variants.random import Random, RandomC


def entrypoint():
    spec = importlib.util.spec_from_file_location("bsbench_execution", Path("scripts/run_bsbench_sweep.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def rating():
    return {"evidence": "B names the error.", "on_axis_A": 0.0, "on_axis_B": 2.0, "off_axis_A": 0.0, "off_axis_B": 0.1}


def request():
    row = {"question_id": "q1", "question_number": 1, "prompt": "Question?", "nonsensical_element": "False premise", "method": "random", "magnitude": 0.3, "side": "-C", "random_seed": 0, "bare": "Base.", "steered": "Changed."}
    return numbered_requests([row], sweep.JUDGE_MODEL, "https://example.invalid")[0]


def test_directed_effect_and_signed_axis_are_distinct():
    for side, signed in (("+C", 2), ("-C", -2)):
        scored = score_pair(rating(), "AB", side)
        assert scored["directed_intended_effect"] == 2
        assert scored["signed_axis_effect"] == signed


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -6, 6])
def test_strict_judge_rating(value):
    with pytest.raises(ValueError):
        validate_judgment(rating() | {"on_axis_A": value}, FORMAT["json_schema"]["schema"])


@pytest.mark.parametrize("value", ["", " " * 3, "a" * 401])
def test_strict_judge_evidence(value):
    with pytest.raises(ValueError):
        validate_judgment(rating() | {"evidence": value}, FORMAT["json_schema"]["schema"])


def test_nested_blind_schema():
    schema = BLIND_FORMAT["json_schema"]["schema"]
    change = {"concept": "stance", "description": "opposes", "evidence_A": "yes", "evidence_B": "no", "change_B_minus_A": 1.5}
    validate_judgment({"summary": "Different stance", "changes": [change]}, schema)
    for invalid in ([change] * 4, [{**change, "change_B_minus_A": True}], [{**change, "extra": "not permitted"}]):
        with pytest.raises(ValueError):
            validate_judgment({"summary": "Different", "changes": invalid}, schema)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -1])
def test_costs_fail_closed(tmp_path, value):
    ledger = tmp_path / "costs.jsonl"
    with pytest.raises(ValueError):
        reserve(ledger, "bad", value)
    reservation = reserve(ledger, "good", 1.0)
    before = ledger.read_bytes()
    with pytest.raises(ValueError):
        settle(ledger, reservation, value)
    assert ledger.read_bytes() == before


def test_five_random_seeds_have_distinct_vectors_and_comparisons(tmp_path, monkeypatch):
    hashes = []
    for seed in sweep.RANDOM_SEEDS:
        acts = {7: torch.zeros(1, 32)}
        vector = Random.extract(acts, acts, RandomC(seed=seed))[7]["stacked"]["v"]
        hashes.append(hashlib.sha256(vector.numpy().tobytes()).hexdigest())
    assert len(hashes) == len(set(hashes)) == 5
    ids = {comparison_id({"question_id": "q", "method": "random", "magnitude": magnitude, "side": side, "random_seed": seed}) for seed in sweep.RANDOM_SEEDS for magnitude in (0.2, 0.4) for side in ("+C", "-C")}
    assert len(ids) == 20
    cli = entrypoint()
    calls = []
    def run_condition(*args, **kwargs):
        seed = kwargs["random_seed"]
        calls.append(seed)
        return {"paid_execution_enabled": False, "random_seed": seed, "candidate": {"vector_sha256": hashes[seed]}}
    monkeypatch.setattr(cli, "run_condition", run_condition)
    result = cli.run_full_sweep(tmp_path, tmp_path / "costs.jsonl", model={"id": "fake", "judge_model": "fake"}, rows=[], backend=object(), prompt_spec={"test": True}, judge=type("Judge", (), {"endpoint": "fake"})(), methods=("random",))
    assert calls == list(sweep.RANDOM_SEEDS)
    assert list(result["conditions"]) == ["random"]
    assert [replica["random_seed"] for replica in result["conditions"]["random"]["replicates"]] == calls


def test_rates_do_not_invalidate_judgment_and_attempts_reserve_on_demand(tmp_path, monkeypatch):
    monkeypatch.setattr(adapters, "JUDGE_INPUT_USD_PER_MTOKEN", .04)
    monkeypatch.setattr(adapters, "JUDGE_OUTPUT_USD_PER_MTOKEN", .64)
    calls = []
    def call(payload):
        calls.append(payload)
        return rating() | {"_remote_cost_usd": 0.0001}
    ledger = tmp_path / "ledger.jsonl"
    judge = LocalJudgeAdapter(call, "https://example.invalid", RunGate(True, {"total_upper_usd": 1.0, "limit_usd": 50.0}), root=tmp_path, ledger=ledger)
    judge.complete([request()])
    before = ledger.read_bytes()
    monkeypatch.setattr(adapters, "JUDGE_OUTPUT_USD_PER_MTOKEN", 1.0)
    judge.complete([request()])
    assert len(calls) == 1 and ledger.read_bytes() == before
    assert [json.loads(line)["event"] for line in before.splitlines()] == ["reserved", "settled"]


@pytest.mark.parametrize("cost", [None, True, float("nan"), float("inf"), -1])
def test_missing_or_invalid_provider_cost_uses_upper(tmp_path, monkeypatch, cost):
    monkeypatch.setattr(adapters, "JUDGE_INPUT_USD_PER_MTOKEN", .04)
    monkeypatch.setattr(adapters, "JUDGE_OUTPUT_USD_PER_MTOKEN", .64)
    ledger = tmp_path / "ledger.jsonl"
    judge = LocalJudgeAdapter(lambda _: rating() | {"_remote_cost_usd": cost}, "https://example.invalid", RunGate(True, {"total_upper_usd": 1., "limit_usd": 50.}), root=tmp_path, ledger=ledger)
    judge.complete([request()])
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reserved", "estimated_at_reservation_upper"]
    assert committed(ledger) == rows[0]["upper_usd"]


def test_six_overlapping_audited_calls_have_unique_evidence(tmp_path, monkeypatch):
    cli = entrypoint()
    barrier = threading.Barrier(6)
    def remote(_):
        barrier.wait(timeout=10)
        return rating()
    monkeypatch.setattr(cli, "openrouter_request_callback", lambda **_: remote)
    call = cli.audited_openrouter_request_callback(endpoint="https://example.invalid", api_key="fake", evidence_root=tmp_path)
    with ThreadPoolExecutor(max_workers=6) as executor:
        responses = list(executor.map(call, [request()["payload"]] * 6))
    evidence = [json.loads(path.read_text()) for path in tmp_path.glob("*.json")]
    assert len(responses) == len(evidence) == len({row["attempt"] for row in evidence}) == 6
    assert all(row["enforced_wait_seconds"] == 0 for row in evidence)


def test_parse_failure_preserves_provider_termination_metadata(monkeypatch):
    body = {"id": "diagnostic-response", "provider": "test-provider", "model": sweep.JUDGE_MODEL, "usage": {"prompt_tokens": 123, "completion_tokens": 1024, "cost": 0.0001}, "error": {"code": "diagnostic"}, "choices": [{"finish_reason": "length", "native_finish_reason": "max_tokens", "message": {"content": '{"evidence":"unfinished', "reasoning": None}}]}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *_): return False
        def read(self): return json.dumps(body).encode()
    captured = []
    def urlopen(sent, **_):
        captured.append(json.loads(sent.data))
        return Response()
    monkeypatch.setattr(adapters, "urlopen", urlopen)
    payload = request()["payload"]
    with pytest.raises(adapters.OpenRouterResponseParseError) as error:
        adapters.openrouter_request_callback(endpoint="https://example.invalid", api_key="test")(payload)
    assert captured == [payload]
    metadata = error.value.evidence["response_metadata"]
    assert metadata["provider"] == body["provider"]
    assert metadata["usage"] == body["usage"]
    assert metadata["error"] == body["error"]
    assert metadata["choices"][0]["finish_reason"] == "length"
    assert metadata["choices"][0]["native_finish_reason"] == "max_tokens"
    assert entrypoint()._redact(metadata)["usage"] == body["usage"]


def test_signed_budget_counts_match_five_seed_scope():
    final = sweep.phase_b_budget_stages()
    assert len([stage for stage in final if stage["runner"] == "modal_gpu"]) == 10
    assert {stage["item_count"] for stage in final if stage["stage"] == "final-generation"} == {168}
    assert {stage["item_count"] for stage in final if stage["runner"] == "local_judge_api"} == {120}
    assert sweep._request_counts(list(final)) == {"target_aware": 4800, "blind": 2400, "persona_validation": 0}
    assert ("candidate-blind", "local_judge_api") not in sweep.condition_stages("mean_diff")
