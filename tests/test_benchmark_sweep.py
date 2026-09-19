from pathlib import Path

import pytest

from steering_lite.benchmark.sweep import (
    cached_dry_stage,
    cost_estimate,
    dry_manifest,
    preflight_budget,
    reserve_budget,
)


def test_dry_manifest_has_numbered_conditions_routes_budget_and_cache_reuse(tmp_path: Path):
    first = dry_manifest(tmp_path, "Qwen/Qwen3.5-4B")
    second = dry_manifest(tmp_path, "Qwen/Qwen3.5-4B")
    estimate = first["cost_estimate"]
    assert [row["question_number"] for row in first["questions"]] == list(range(1, 21))
    assert len(first["conditions"]) == 8
    assert {stage["runner"] for stage in first["stages"]} == {"modal_gpu", "local", "local_judge_api"}
    assert not any(stage["reused"] for stage in first["stages"])
    assert all(stage["reused"] for stage in second["stages"])
    assert estimate["quantities"]["gpu_stages"] == 26
    assert estimate["quantities"]["requests"] == {"target_aware": 320, "blind": 320, "persona_validation": 12}
    assert estimate["judge_model"] == first["judge_model"] == "deepseek/deepseek-chat"
    assert estimate["total_upper_usd"] < 50.0
    assert estimate["expected_work"][0]["rate_source"].startswith("https://modal.com/pricing")
    assert estimate["expected_work"][-1]["rate_source"].startswith("https://openrouter.ai/")
    assert first["paid_execution_enabled"] is False
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "cost-estimate.json").exists()
    assert (tmp_path / "dry-plan-cache").exists()
    assert not (tmp_path / "cache").exists()


def test_stage_cache_invalidates_model_data_method_config_prompt_and_code(tmp_path: Path):
    inputs = {"stage": "generation", "runner": "modal_gpu", "model": {"id": "a"}, "data": {"sha256": "a"}, "method": "mean_diff", "config": {"coefficient": 1.0}, "prompts": ["a"], "code": "a"}
    assert not cached_dry_stage(tmp_path, **inputs)["reused"]
    assert cached_dry_stage(tmp_path, **inputs)["reused"]
    for changed in ({"model": {"id": "b"}}, {"data": {"sha256": "b"}}, {"method": "pca"}, {"config": {"coefficient": 2.0}}, {"prompts": ["b"]}, {"code": "b"}):
        assert not cached_dry_stage(tmp_path, **(inputs | changed))["reused"]


def test_costs_follow_stage_and_request_quantities_and_exact_fifty_fails(tmp_path: Path):
    manifest = dry_manifest(tmp_path)
    full = cost_estimate(manifest["stages"], len(manifest["questions"]))
    fewer_gpu = cost_estimate([stage for stage in manifest["stages"] if stage["runner"] != "modal_gpu"], len(manifest["questions"]))
    fewer_requests = cost_estimate([stage for stage in manifest["stages"] if stage["stage"] != "blind-requests"], len(manifest["questions"]))
    assert fewer_gpu["total_upper_usd"] < full["total_upper_usd"]
    assert fewer_requests["quantities"]["input_tokens"] < full["quantities"]["input_tokens"]
    assert fewer_requests["total_upper_usd"] < full["total_upper_usd"]
    assert len(reserve_budget(tmp_path / "costs.jsonl", full)) == 3
    at_limit = {"total_upper_usd": 50.0, "planned_reservations": [{"kind": "first", "upper_usd": 10.0}, {"kind": "second", "upper_usd": 40.0}]}
    with pytest.raises(RuntimeError, match="at or above"):
        reserve_budget(tmp_path / "at-limit.jsonl", at_limit)
    assert not (tmp_path / "at-limit.jsonl").exists()
