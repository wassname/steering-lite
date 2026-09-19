from pathlib import Path

import pytest

from steering_lite.benchmark.cache import content_key
from steering_lite.benchmark.dose_search import CALIBRATION_CASE, Case, TRANSFER_CASES
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.sweep import (
    BSBENCH_PERSONAS,
    BSBENCH_PERSONA_N_PAIRS,
    BSBENCH_PERSONA_SEED,
    BSBENCH_PERSONA_TEMPLATE,
    BSBENCH_PERSONA_THINKING,
    cached_dry_stage,
    cost_estimate,
    dry_manifest,
    final_stages,
    preflight_budget,
    reserve_budget,
)


def _method_stages(manifest: dict, method: str) -> list[dict]:
    return [stage for stage in manifest["stages"] if stage["method"] == method]


def _actual_transfer_cases() -> tuple[Case, ...]:
    return (
        Case("bsbench-v2-heldout-transfer", "bsbench-v2-heldout", ("BSV2-H-001",)),
        Case("other-dataset-transfer", "other-dataset", ("OTHER-001", "OTHER-002")),
    )


def _final_inputs() -> dict:
    transfer_cases = _actual_transfer_cases()
    return {
        "method": "vjp_cache",
        "vector_sha256": "vector-a",
        "observed": [
            {"coefficient": 0.4, "useful": True, "coherent": True, "provenance": "judge-b"},
            {"coefficient": -0.2, "useful": True, "coherent": True, "provenance": "judge-a"},
        ],
        "case_prompts": {
            transfer_cases[0].case_id: ["heldout prompt"],
            transfer_cases[1].case_id: ["other prompt one", "other prompt two"],
        },
        "prompt_spec": {"template": "Answer plainly.", "max_new_tokens": 128},
        "transfer_cases": transfer_cases,
    }


def test_dry_manifest_has_exact_phase_a_graph_identities_counts_and_cache_reuse(tmp_path: Path):
    first = dry_manifest(tmp_path, "Qwen/Qwen3.5-4B")
    second = dry_manifest(tmp_path, "Qwen/Qwen3.5-4B")
    rows = read_dev_cohort()
    calibration_prompts = [
        next(row["prompt"] for row in rows if row["question_id"] == prompt_id)
        for prompt_id in CALIBRATION_CASE.prompt_ids
    ]
    estimate = first["cost_estimate"]

    assert [row["question_number"] for row in first["questions"]] == list(range(1, 21))
    assert first["conditions"] == list(METHODS)
    assert not any(stage["reused"] for stage in first["stages"])
    assert all(stage["reused"] for stage in second["stages"])

    for method in ("bare", "prompting"):
        stages = _method_stages(first, method)
        assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in stages] == [
            ("generation", "modal_gpu", 20),
            ("generation-health", "local", 20),
            ("target-aware-requests", "local_judge_api", 20),
            ("blind-requests", "local_judge_api", 20),
        ]
        assert all(stage["config"]["persona_source"] is None for stage in stages)

    vector_methods = set(METHODS) - {"bare", "prompting"}
    assert len(vector_methods) == 6
    expected_persona_source = {
        "pairs": [list(pair) for pair in BSBENCH_PERSONAS],
        "template": BSBENCH_PERSONA_TEMPLATE,
        "seed": BSBENCH_PERSONA_SEED,
        "n_pairs": BSBENCH_PERSONA_N_PAIRS,
        "thinking": BSBENCH_PERSONA_THINKING,
    }
    for method in vector_methods:
        stages = _method_stages(first, method)
        assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in stages] == [
            ("calibration-candidates", "modal_gpu", 4),
            ("candidate-health", "local", 4),
            ("candidate-aware", "local_judge_api", 4),
            ("candidate-blind", "local_judge_api", 4),
        ]
        assert {stage["prompts_sha256"] for stage in stages} == {
            content_key({"prompts": calibration_prompts})
        }
        assert all(stage["config"]["calibration_case"]["prompt_ids"] == list(CALIBRATION_CASE.prompt_ids) for stage in stages)
        assert all(stage["config"]["calibration_prompts_sha256"] == content_key({"prompts": calibration_prompts}) for stage in stages)
        assert all(stage["config"]["persona_source"] == expected_persona_source for stage in stages)
        assert all(stage["config"]["persona_source_sha256"] == content_key(expected_persona_source) for stage in stages)

    assert estimate["quantities"]["gpu_stages"] == 8
    assert estimate["quantities"]["requests"] == {
        "target_aware": 128,
        "blind": 128,
        "persona_validation": 12,
    }
    assert estimate["judge_model"] == first["judge_model"] == "deepseek/deepseek-chat"
    assert estimate["total_upper_usd"] < 50.0
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


def test_final_stages_carry_complete_data_flow_and_stable_identity():
    inputs = _final_inputs()
    stages = final_stages(**inputs)
    reversed_inputs = inputs | {
        "observed": list(reversed(inputs["observed"])),
        "case_prompts": dict(reversed(list(inputs["case_prompts"].items()))),
    }
    reordered = final_stages(**reversed_inputs)

    assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in stages] == [
        ("final-generation", "modal_gpu", 3),
        ("final-health", "local", 3),
        ("final-aware", "local_judge_api", 3),
        ("final-blind", "local_judge_api", 3),
    ]
    assert stages == reordered
    assert stages[0]["case_prompts"] == inputs["case_prompts"]
    assert all("case_prompts" not in stage for stage in stages[1:])
    assert all(stage["input_stage"] == "final-generation" for stage in stages[1:])
    config = stages[0]["config"]
    assert config["vector_sha256"] == "vector-a"
    assert config["candidate_coefficients"] == [-0.2, 0.4]
    assert config["prompt_spec"] == inputs["prompt_spec"]
    assert config["prompt_spec_sha256"] == content_key(inputs["prompt_spec"])
    assert config["case_prompt_hashes"] == {
        case_id: content_key({"prompts": prompts})
        for case_id, prompts in inputs["case_prompts"].items()
    }
    assert all(stage["config"] == config for stage in stages)


def test_final_stage_identity_invalidates_each_input_independently():
    inputs = _final_inputs()
    baseline = final_stages(**inputs)[0]["config"]
    changes = (
        {"vector_sha256": "vector-b"},
        {"observed": [{**inputs["observed"][0], "provenance": "different"}, inputs["observed"][1]]},
        {"observed": [{**inputs["observed"][0], "coefficient": 0.5}, inputs["observed"][1]]},
        {"case_prompts": inputs["case_prompts"] | {"other-dataset-transfer": ["changed prompt"]}},
        {"prompt_spec": inputs["prompt_spec"] | {"template": "Answer directly."}},
    )
    for changed in changes:
        assert final_stages(**(inputs | changed))[0]["config"] != baseline


def test_final_stages_reject_missing_or_placeholder_inputs_before_dispatch():
    inputs = _final_inputs()
    for changed, message in (
        ({"vector_sha256": ""}, "vector"),
        ({"observed": []}, "observed"),
        ({"case_prompts": {}}, "prompts"),
        ({"prompt_spec": {}}, "prompt spec"),
        ({"case_prompts": {"bsbench-v2-heldout-transfer": ["only one"]}}, "complete prompts"),
    ):
        with pytest.raises(ValueError, match=message):
            final_stages(**(inputs | changed))

    with pytest.raises(ValueError, match="placeholder"):
        final_stages(
            **(inputs | {
                "transfer_cases": TRANSFER_CASES,
                "case_prompts": {case.case_id: ["loadable"] for case in TRANSFER_CASES},
            })
        )


def test_costs_use_per_stage_counts_and_blind_filtering(tmp_path: Path):
    manifest = dry_manifest(tmp_path)
    full = cost_estimate(manifest["stages"])
    no_gpu = cost_estimate([stage for stage in manifest["stages"] if stage["runner"] != "modal_gpu"])
    no_blind = cost_estimate([stage for stage in manifest["stages"] if "blind" not in stage["stage"]])

    assert no_gpu["total_upper_usd"] < full["total_upper_usd"]
    assert no_blind["quantities"]["requests"] == {
        "target_aware": 128,
        "blind": 0,
        "persona_validation": 12,
    }
    assert no_blind["quantities"]["input_tokens"] < full["quantities"]["input_tokens"]
    assert no_blind["total_upper_usd"] < full["total_upper_usd"]
    with pytest.raises(KeyError, match="item_count"):
        cost_estimate([{"stage": "final-blind", "runner": "local_judge_api"}])

    assert len(reserve_budget(tmp_path / "costs.jsonl", full)) == 3
    at_limit = {"total_upper_usd": 50.0, "planned_reservations": [{"kind": "first", "upper_usd": 10.0}, {"kind": "second", "upper_usd": 40.0}]}
    with pytest.raises(RuntimeError, match="at or above"):
        reserve_budget(tmp_path / "at-limit.jsonl", at_limit)
    assert not (tmp_path / "at-limit.jsonl").exists()
