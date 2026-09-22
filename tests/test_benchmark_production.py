import importlib.util
import io
import json
import os
from contextlib import contextmanager
from urllib.error import HTTPError
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from steering_lite.benchmark.cache import settle_receipt
from steering_lite.benchmark.production import persist_local_judge_work, run_stages


class FakeModal:
    def __init__(self, ledger: Path):
        self.ledger = ledger
        self.calls = []

    def gpu(self, **kwargs):
        assert self.ledger.exists(), "dispatch happened before reservation"
        self.calls.append(kwargs["stage"])
        return {"actual_usd": 0.25, "stage": kwargs["stage"], "method": kwargs["method"]}


def test_production_stages_reserve_before_dispatch_reuse_and_invalidate(tmp_path: Path):
    ledger = tmp_path / "costs.jsonl"
    backend = FakeModal(ledger)
    common = {"model": {"id": "Qwen/Qwen3.5-4B"}, "data": {"sha256": "dev"}, "method": "vjp_cache", "prompts": ["one"]}
    stages = [{**common, "stage": "extract", "config": {"upper_usd": 0.5, "step": 1}}, {**common, "stage": "generate", "config": {"upper_usd": 0.5, "step": 2}}]
    first = run_stages(tmp_path, ledger, stages, backend)
    assert backend.calls == ["extract", "generate"]
    assert not any(row["reused"] for row in first)
    assert all(row["reused"] for row in run_stages(tmp_path, ledger, stages, backend))
    changed = [{**stages[0], "config": {"upper_usd": 0.5, "step": 3}}]
    assert not run_stages(tmp_path, ledger, changed, backend)[0]["reused"]


def test_modal_pending_receipt_is_cached_estimated_and_reconciled_later(tmp_path: Path):
    ledger = tmp_path / "costs.jsonl"
    calls = []

    class PendingModal:
        def gpu(self, **kwargs):
            calls.append(kwargs["stage"])
            return {"stage": kwargs["stage"], "cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": 1.0}}}

    stage = {"model": {"id": "fake"}, "data": {"sha256": "dev"}, "method": "bare", "stage": "generation", "config": {"upper_usd": 0.5}, "prompts": ["one"]}
    first, = run_stages(tmp_path, ledger, [stage], PendingModal())
    assert first["reused"] is False and "actual_usd" not in first
    assert calls == ["generation"]
    assert run_stages(tmp_path, ledger, [stage], PendingModal())[0]["reused"] is True
    assert calls == ["generation"]
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["reserved", "estimated_at_reservation_upper"]
    assert rows[-1]["estimated_usd"] == 0.5 and "actual_usd" not in rows[-1]
    next_stage = {**stage, "stage": "another", "config": {"upper_usd": 0.5}}
    assert run_stages(tmp_path, ledger, [next_stage], PendingModal())[0]["reused"] is False
    settle_receipt(ledger, first["reservation"], 0.25, {"provider": "Modal", "receipt_id": "later"})
    reconciled = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in reconciled][:4] == ["reserved", "estimated_at_reservation_upper", "reserved", "estimated_at_reservation_upper"]
    assert [row["event"] for row in reconciled][-2:] == ["settled", "receipt_imported"]


def test_cli_imports_modal_receipt_without_remote_dispatch(tmp_path: Path):
    ledger = tmp_path / "costs.jsonl"

    class PendingModal:
        def gpu(self, **_kwargs):
            return {"cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": 1.0}}}

    stage = {"model": {"id": "fake"}, "data": {"sha256": "dev"}, "method": "bare", "stage": "generation", "config": {"upper_usd": 0.5}, "prompts": ["one"]}
    result, = run_stages(tmp_path, ledger, [stage], PendingModal())
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps({"reservation": result["reservation"], "actual_usd": 0.25, "receipt": {"provider": "Modal", "receipt_id": "cli"}}))
    command = [sys.executable, "scripts/run_bsbench_sweep.py", "--import-receipt", str(receipt_path), "--ledger", str(ledger)]
    output = subprocess.run(command, cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True)
    assert json.loads(output.stdout)["mode"] == "receipt-import"
    assert [json.loads(line)["event"] for line in ledger.read_text().splitlines()][-2:] == ["settled", "receipt_imported"]


def test_sweep_recipe_exports_project_env_without_printing_key(tmp_path: Path):
    root = Path(__file__).parents[1]
    env_file = tmp_path / "project.env"
    sentinel = "sentinel-openrouter-key"
    env_file.write_text(f"OPENROUTER_API_KEY={sentinel}\n")
    environment = os.environ.copy()
    environment.pop("OPENROUTER_API_KEY", None)
    command = [
        "just", "sweep", "--check-openrouter-env", "Qwen/Qwen3.5-4B",
        str(tmp_path / "check"), str(env_file),
    ]
    checked = subprocess.run(command, cwd=root, env=environment, check=True, capture_output=True, text=True)
    assert json.loads(checked.stdout.splitlines()[-1]) == {
        "mode": "check-openrouter-env", "openrouter_api_key_present": True,
    }
    assert sentinel not in checked.stdout + checked.stderr
    absent = subprocess.run(
        ["just", "sweep", "--dry-run", "Qwen/Qwen3.5-4B", str(tmp_path / "dry"), str(tmp_path / "missing.env")],
        cwd=root, env=environment, check=False, capture_output=True, text=True,
    )
    assert absent.returncode == 1
    assert "judge pricing is unsourced" in absent.stderr
    assert sentinel not in absent.stdout + absent.stderr


def test_modal_receipt_overage_is_detected_after_estimate(tmp_path: Path):
    ledger = tmp_path / "costs.jsonl"

    class PendingModal:
        def gpu(self, **kwargs):
            return {"cost_receipt": {"status": "pending", "provider": "Modal", "usage": {"elapsed_seconds": 1.0}}}

    stage = {"model": {"id": "fake"}, "data": {"sha256": "dev"}, "method": "bare", "stage": "generation", "config": {"upper_usd": 0.5}, "prompts": ["one"]}
    result, = run_stages(tmp_path, ledger, [stage], PendingModal())
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        settle_receipt(ledger, result["reservation"], 0.6, {"provider": "Modal", "receipt_id": "over"})
    assert [json.loads(line)["event"] for line in ledger.read_text().splitlines()] == ["reserved", "estimated_at_reservation_upper", "settled", "overage"]


def test_local_judge_work_persists_persona_checks_and_both_request_kinds(tmp_path: Path):
    row = {"question_id": "BSV2-001", "question_number": 1, "prompt": "Is this real?", "nonsensical_element": "No.", "bare": "No, it is not real.", "steered": "No, it is not real.", "side": "+C", "method": "vjp_cache", "coefficient": 0.2}
    example = {"pair_id": "pair-1", "positive_persona": "candid", "negative_persona": "agreeable", "positive": "I disagree. Shared.", "negative": "I agree. Shared.", "shared_suffix": "Shared."}
    result = persist_local_judge_work(tmp_path, model={"id": "Qwen/Qwen3.5-4B", "judge_model": "deepseek/deepseek-v4-flash-0731"}, data={"sha256": "dev"}, rows=[row], persona_examples=[example], endpoint="offline")
    assert result["persona_checks"][0]["status"] == "local_structural_checks"
    assert len(result["requests"]) == 6
    assert {record["blind"] for record in result["requests"]} == {False, True}


def test_recorded_run_reuses_temp_ledger_and_rejects_wrong_model(tmp_path: Path):
    root = Path(__file__).parents[1]
    command = [sys.executable, "scripts/run_bsbench_sweep.py", "--run", "--backend", "recorded", "--out", str(tmp_path / "out"), "--ledger", str(tmp_path / "ledger.jsonl")]
    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    second = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    assert "cache hit generation" in second.stderr
    assert len((root / "outputs/bsbench-smoke/costs.jsonl").read_text().splitlines()) == 1
    wrong = command[:-2] + ["--model", "wrong/model"]
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(wrong, cwd=root, check=True, capture_output=True, text=True)


def _sweep_script():
    path = Path(__file__).parents[1] / "scripts" / "run_bsbench_sweep.py"
    spec = importlib.util.spec_from_file_location("bsbench_sweep_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_openrouter_http_error_evidence_preserves_retry_metadata_without_key(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()

    def fail(_payload):
        raise HTTPError(
            "https://openrouter.ai/api/v1/chat/completions",
            429,
            "Too Many Requests",
            {"Retry-After": "60", "X-RateLimit-Remaining": "0"},
            io.BytesIO(b'{"error":{"message":"rate limited","key":"must-not-persist"}}'),
        )

    monkeypatch.setattr(sweep, "openrouter_request_callback", lambda **_kwargs: fail)
    request_call = sweep.audited_openrouter_request_callback(
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key="must-not-persist",
        evidence_root=tmp_path,
    )
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "private prompt"}],
        "response_format": {"json_schema": {"name": "persona_pair_validation"}},
    }
    with pytest.raises(HTTPError, match="Too Many Requests"):
        request_call(payload)

    evidence, = tmp_path.glob("*.json")
    record = json.loads(evidence.read_text())
    assert record["status"] == 429
    assert record["headers"] == {"retry-after": "60", "x-ratelimit-remaining": "0"}
    assert record["body"] == {"error": {"message": "rate limited", "key": "<redacted>"}}
    assert "must-not-persist" not in evidence.read_text()
    assert "private prompt" not in evidence.read_text()


def test_openrouter_no_response_evidence_has_request_identity_without_payload(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()
    monkeypatch.setattr(sweep, "openrouter_request_callback", lambda **_kwargs: lambda _payload: (_ for _ in ()).throw(TimeoutError("private prompt must not persist")))
    clock = iter((1.0, 1.0, 4.0))
    monkeypatch.setattr(sweep.time, "monotonic", lambda: next(clock))
    request_call = sweep.audited_openrouter_request_callback(
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key="must-not-persist",
        evidence_root=tmp_path,
    )
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": "private prompt"}],
        "response_format": {"json_schema": {"name": "blind_change_description"}},
    }
    with pytest.raises(TimeoutError):
        request_call(payload)

    evidence, = tmp_path.glob("*.json")
    record = json.loads(evidence.read_text())
    assert record["schema"] == "bsbench-openrouter-no-response-v1"
    assert record["exception_type"] == "TimeoutError"
    assert record["elapsed_seconds"] == 3.0
    assert record["response_schema"] == "blind_change_description"
    assert "must-not-persist" not in evidence.read_text()
    assert "private prompt" not in evidence.read_text()


def test_openrouter_read_timeout_overrides_adapter_default(monkeypatch):
    sweep = _sweep_script()
    import steering_lite.benchmark.adapters as adapters

    seen = []

    def urlopen(*_args, **kwargs):
        seen.append(kwargs["timeout"])

    monkeypatch.setattr(adapters, "urlopen", urlopen)
    with sweep._openrouter_read_timeout(180.0):
        adapters.urlopen("request", timeout=90)
    assert seen == [180.0]
    assert adapters.urlopen("request", timeout=90) is None
    assert seen == [180.0, 90]


def test_openrouter_callback_records_dispatch_starts_at_least_ten_seconds_apart(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()
    clock = [0.0]
    dispatch_starts = []
    sleeps = []

    def request(_payload):
        dispatch_starts.append(clock[0])
        clock[0] += 0.25
        return {"ok": True}

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(sweep, "openrouter_request_callback", lambda **_kwargs: request)
    monkeypatch.setattr(sweep.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(sweep.time, "sleep", sleep)
    request_call = sweep.audited_openrouter_request_callback(
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        api_key="must-not-persist",
        evidence_root=tmp_path,
        min_interval_seconds=10.0,
    )
    payload = {"model": "deepseek/deepseek-chat", "response_format": {"json_schema": {"name": "persona_pair_validation"}}}
    assert request_call(payload) == request_call(payload) == {"ok": True}

    assert dispatch_starts[1] - dispatch_starts[0] >= 10.0
    assert sleeps == [9.75]
    timing_records = [json.loads(path.read_text()) for path in sorted(tmp_path.glob("*.json"))]
    assert [record["outcome"] for record in timing_records] == ["success", "success"]
    assert [record["enforced_wait_seconds"] for record in timing_records] == [0.0, 9.75]
    assert [record["elapsed_seconds"] for record in timing_records] == [0.25, 0.25]
    assert all(record["dispatch_started_at"] and record["response_finished_at"] for record in timing_records)


def test_openrouter_metadata_redacts_key_fields(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()

    class Response:
        status = 200
        headers = {"Content-Type": "application/json", "X-RateLimit-Limit": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"data":{"limit_remaining":4.5,"key_hash":"must-not-persist","user_id":"must-not-persist"}}'

    monkeypatch.setattr(sweep, "urlopen", lambda *_args, **_kwargs: Response())
    metadata = sweep.openrouter_metadata(api_key="must-not-persist")
    assert {name: value["status"] for name, value in metadata["endpoints"].items()} == {"key": 200, "credits": 200}
    assert all(value["body"]["data"]["key_hash"] == "<redacted>" for value in metadata["endpoints"].values())
    assert all(value["body"]["data"]["user_id"] == "<redacted>" for value in metadata["endpoints"].values())


def test_real_sweep_keeps_one_modal_lifecycle_across_gpu_stages(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()
    events = []
    remote_calls = []
    read_timeouts = []

    @contextmanager
    def read_timeout(seconds):
        read_timeouts.append(seconds)
        yield

    monkeypatch.setattr(sweep, "_openrouter_read_timeout", read_timeout)

    class App:
        active = False

        def run(self):
            events.append("run")
            return self

        def __enter__(self):
            assert not self.active
            self.active = True
            events.append("enter")
            return self

        def __exit__(self, *_args):
            assert self.active
            self.active = False
            events.append("exit")

    app = App()
    modal_module = ModuleType("run_bsbench_modal")
    modal_module.app = app

    def remote_stage_call(_model, *, explicit_run, budget_preflight):
        assert app.active and explicit_run
        assert budget_preflight == {"total_upper_usd": 1.0, "limit_usd": 50.0}

        def call(**kwargs):
            remote_calls.append((app.active, kwargs["method"]))
            return {"actual_usd": 0.0}

        return call

    modal_module.remote_stage_call = remote_stage_call
    monkeypatch.setitem(sys.modules, "run_bsbench_modal", modal_module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel")
    monkeypatch.setattr(sweep, "dry_manifest", lambda *_args, **_kwargs: {"cost_estimate": {"total_upper_usd": 1.0, "limit_usd": 50.0}})
    monkeypatch.setattr(sweep, "read_dev_cohort", lambda: [])

    def run_two_stages(_root, _ledger, *, backend, **_kwargs):
        backend.gpu(stage="generation", method="bare", config={}, prompts=["one"])
        backend.gpu(stage="generation", method="prompting", config={}, prompts=["two"])
        return {"ok": True}

    monkeypatch.setattr(sweep, "run_full_sweep", run_two_stages)
    monkeypatch.setattr(sys, "argv", ["run_bsbench_sweep.py", "--run", "--backend", "real", "--openrouter-read-timeout", "300", "--out", str(tmp_path / "out")])
    sweep.main()

    assert events == ["run", "enter", "exit"]
    assert read_timeouts == [300.0]
    assert remote_calls == [(True, "bare"), (True, "prompting")]


def test_real_sweep_lifecycle_failure_precedes_production_reservation(tmp_path: Path, monkeypatch):
    sweep = _sweep_script()

    class FailingApp:
        def run(self):
            return self

        def __enter__(self):
            raise RuntimeError("unusable Modal lifecycle")

        def __exit__(self, *_args):
            raise AssertionError("unreachable")

    modal_module = ModuleType("run_bsbench_modal")
    modal_module.app = FailingApp()
    modal_module.remote_stage_call = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unreachable"))
    monkeypatch.setitem(sys.modules, "run_bsbench_modal", modal_module)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sentinel")
    monkeypatch.setattr(sweep, "dry_manifest", lambda *_args, **_kwargs: {"cost_estimate": {"total_upper_usd": 1.0, "limit_usd": 50.0}})
    monkeypatch.setattr(sweep, "run_full_sweep", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unreachable")))
    ledger = tmp_path / "costs.jsonl"
    monkeypatch.setattr(sys, "argv", ["run_bsbench_sweep.py", "--run", "--backend", "real", "--out", str(tmp_path / "out"), "--ledger", str(ledger)])

    with pytest.raises(RuntimeError, match="unusable Modal lifecycle"):
        sweep.main()
    assert not ledger.exists()
