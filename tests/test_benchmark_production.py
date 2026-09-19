from pathlib import Path
import subprocess
import sys

import pytest

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


def test_local_judge_work_persists_persona_checks_and_both_request_kinds(tmp_path: Path):
    row = {"question_id": "BSV2-001", "question_number": 1, "prompt": "Is this real?", "nonsensical_element": "No.", "bare": "No, it is not real.", "steered": "No, it is not real.", "side": "+C", "method": "vjp_cache", "coefficient": 0.2}
    example = {"pair_id": "pair-1", "positive_persona": "candid", "negative_persona": "agreeable", "positive": "I disagree. Shared.", "negative": "I agree. Shared.", "shared_suffix": "Shared."}
    result = persist_local_judge_work(tmp_path, model={"id": "Qwen/Qwen3.5-4B", "judge_model": "deepseek/deepseek-chat"}, data={"sha256": "dev"}, rows=[row], persona_examples=[example], endpoint="offline")
    assert result["persona_checks"][0]["status"] == "local_structural_checks"
    assert len(result["requests"]) == 4
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
