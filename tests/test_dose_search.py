from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.dose_search import (
    CALIBRATION_CASE,
    TRANSFER_CASES,
    Case,
    cached_search,
    classify_boundary,
    classify_transfer_boundary,
    final_dose_plan,
    fit_target,
    predict_transfer,
    validate_cases,
)


def observed(coefficient: float, *, useful: bool, coherent: bool, provenance: str) -> dict:
    return {
        "coefficient": coefficient,
        "historical_score_positive": useful,
        "historical_off_axis_within_2_5": coherent,
        "provenance": provenance,
        "generation_health": {"reasons": []},
    }


def metrics(kl_rms: float) -> dict:
    return {
        "kl_rms": kl_rms,
        "rep": 0.0,
        "gen_len": 20,
        "steer_tail": "calibration response",
        "per_t_mean": [0.1, 0.2],
        "per_t_p90": [0.2, 0.3],
        "per_t_p95": [0.2, 0.3],
        "per_t_max": [0.2, 0.3],
        "per_t_n": [1, 1],
    }


def test_explicit_case_manifests_are_disjoint_and_include_other_datasets():
    validate_cases(CALIBRATION_CASE, TRANSFER_CASES)
    assert len(TRANSFER_CASES) == 4
    assert {case.dataset for case in TRANSFER_CASES if case.dataset != "bsbench-v2-heldout"} == {
        "paper-native-false-claim-agreement",
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_cases(Case("cal", "a", ("one",)), (Case("x", "b", ("two",)), Case("y", "c", ("two",))))


def test_fit_target_uses_largest_generation_healthy_candidate_not_behavioral_cutoffs():
    vector = SimpleNamespace(cfg=SimpleNamespace(coeff=0.0))
    calls = []

    def measure(*args, **kwargs):
        calls.append((args, kwargs))
        return metrics(1.25)

    target = fit_target(
        vector,
        object(),
        object(),
        ["calibration prompt"],
        CALIBRATION_CASE,
        [
            observed(0.2, useful=True, coherent=True, provenance="judge-001"),
            observed(0.4, useful=True, coherent=True, provenance="judge-002"),
            observed(0.8, useful=False, coherent=False, provenance="judge-003"),
        ],
        method="vjp_cache",
        model_id="tiny",
        measure_kwargs={"device": "cpu"},
        measure=measure,
    )
    assert vector.cfg.coeff == 0.8
    assert calls[0][1] == {"device": "cpu"}
    assert target["target_stat"] == "kl_rms"
    assert target["target_rms"] == 1.25
    assert target["source"]["provenance"] == "judge-003"
    assert target["calibration_health"]["per_t_p95"] == [0.2, 0.3]


def test_predict_transfer_calls_solver_at_fitted_target_without_behavioral_observations():
    calls = []

    def solver(*_args, **kwargs):
        calls.append(kwargs)
        return 0.7, [
            {"coeff": 0.2, "kl_rms": 0.4},
            {"coeff": 0.7, "kl_rms": 1.25, "final": True},
        ]

    target = {
        "target_id": "target-a",
        "target_stat": "kl_rms",
        "target_rms": 1.25,
        "source": {"method": "vjp_cache", "model": "tiny"},
    }
    record = predict_transfer(
        object(),
        object(),
        object(),
        ["transfer prompt"],
        target,
        TRANSFER_CASES[0],
        bracket=(0.1, 4.0),
        solver_kwargs={"device": "cpu"},
        solver=solver,
    )
    assert calls == [{"target_kl": 1.25, "target_stat": "kl_rms", "bracket": (0.1, 4.0), "device": "cpu"}]
    assert record["predicted_coefficient"] == 0.7
    assert record["search_history"][-1]["final"] is True
    assert "boundary" not in record
    planned_observations = [
        {"case_id": record["case"]["case_id"], "target_id": record["target_id"], "coefficient": coefficient, "useful": True, "coherent": True, "provenance": "transfer-judge", "generation_health": {}}
        for coefficient in (0.56, 0.7, 0.84)
    ]
    classified = classify_transfer_boundary(record, planned_observations)
    assert classified["boundary"] == "measured_generation_health_boundary"
    with pytest.raises(ValueError, match="nearby observed"):
        classify_transfer_boundary(record, [])
    for changed, message in (
        ([{**row, "case_id": "calibration"} for row in planned_observations], "case and target"),
        ([{**row, "case_id": "other-case"} for row in planned_observations], "case and target"),
        ([{key: value for key, value in row.items() if key != "target_id"} for row in planned_observations], "case_id, target_id and coefficient"),
        ([{**row, "target_id": "other-target"} for row in planned_observations], "case and target"),
        (planned_observations[:2], "planned doses"),
        (planned_observations + [{**planned_observations[0], "coefficient": 0.9}], "planned doses"),
    ):
        with pytest.raises(ValueError, match=message):
            classify_transfer_boundary(record, changed)


def test_final_dose_plan_uses_predicted_plus_fixed_nearby_doses():
    plan = final_dose_plan({
        "target_id": "target-a",
        "method": "vjp_cache",
        "case": {"case_id": "transfer-a", "dataset": "synthetic", "prompt_ids": ["SYN-001"]},
        "predicted_coefficient": -0.5,
    })
    assert plan["coefficients"] == [-0.4, -0.5, -0.6]
    with pytest.raises(ValueError, match="non-zero"):
        final_dose_plan({"target_id": "target-a", "method": "vjp_cache", "case": {"case_id": "transfer-a", "dataset": "synthetic", "prompt_ids": ["SYN-001"]}, "predicted_coefficient": 0.0})


def test_boundary_distinguishes_failure_from_solver_limit_including_final_point():
    assert classify_boundary(
        [{"coeff": 4.0, "kl_rms": 0.1, "final": True}],
        [observed(0.5, useful=True, coherent=True, provenance="judge")],
        1.0,
        (0.1, 4.0),
    ) == "search_limit"
    assert classify_boundary(
        [{"coeff": 0.5, "kl_rms": 1.0, "final": True}],
        [
            observed(0.5, useful=True, coherent=True, provenance="judge-good"),
            observed(0.8, useful=False, coherent=False, provenance="judge-still-healthy"),
        ],
        1.0,
        (0.1, 4.0),
    ) == "measured_generation_health_boundary"


def test_cached_search_reuses_identical_inputs_and_invalidates_changes(tmp_path: Path):
    calls = []

    def compute():
        calls.append(None)
        return {"call": len(calls)}

    inputs = {
        "model": {"id": "tiny", "revision": "a"},
        "data": {"sha256": "data-a"},
        "method": "vjp_cache",
        "config": {"target_stat": "kl_rms", "target_rms": 1.25},
        "prompts": ["prompt-a"],
        "compute": compute,
    }
    assert cached_search(tmp_path, **inputs) == {"call": 1}
    assert cached_search(tmp_path, **inputs) == {"call": 1}
    assert cached_search(tmp_path, **(inputs | {"config": {"target_stat": "kl_rms", "target_rms": 1.5}})) == {"call": 2}
    assert cached_search(tmp_path, **(inputs | {"prompts": ["prompt-b"]})) == {"call": 3}
