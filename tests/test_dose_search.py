import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.dose_search import (
    BENCHMARK_KL_SPEC,
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


def observed(magnitude: float, *, useful: bool, coherent: bool, provenance: str, side: str = "+C") -> dict:
    return {
        "magnitude": magnitude,
        "coefficient": magnitude,
        "side": side,
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
        "n_pos": 2,
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
            observed(magnitude, useful=useful, coherent=coherent, provenance=f"judge-{magnitude}-{side}", side=side)
            for magnitude, useful, coherent in ((0.2, True, True), (0.4, True, True), (0.8, False, False))
            for side in ("+C", "-C")
        ],
        method="vjp_cache",
        model_id="tiny",
        kl_spec=BENCHMARK_KL_SPEC,
        measure_kwargs={"device": "cpu"},
        measure=measure,
    )
    assert vector.cfg.coeff == -0.8
    expected_kwargs = {"device": "cpu", "T": 20, "do_sample": True, "seed": 0}
    assert [call[1] for call in calls] == [expected_kwargs, expected_kwargs]
    assert target["target_stat"] == "kl_rms"
    assert target["target_rms"] == 1.25
    assert set(target["source"]["provenance"]) == {"+C", "-C"}
    assert target["signed_components"][0]["health"]["per_t_p95"] == [0.2, 0.3]


def test_predict_transfer_calls_solver_at_fitted_target_without_behavioral_observations():
    calls = []

    def solver(*_args, **kwargs):
        calls.append(kwargs)
        coefficient = 3.0 * kwargs["sign"]
        return coefficient, [
            {"coeff": 0.2 * kwargs["sign"], "kl_rms": 0.4},
            {"coeff": coefficient, "kl_rms": 1.25, "final": True},
        ]

    target = {
        "target_id": "target-a",
        "target_stat": "kl_rms",
        "target_rms": 1.25,
        "kl_spec": BENCHMARK_KL_SPEC,
        "source": {"method": "vjp_cache", "model": "tiny"},
    }
    record = predict_transfer(
        object(),
        object(),
        object(),
        ["transfer prompt"],
        json.loads(json.dumps(target)),
        TRANSFER_CASES[0],
        kl_spec=BENCHMARK_KL_SPEC,
        solver_kwargs={"device": "cpu"},
        solver=solver,
    )
    assert [call["sign"] for call in calls] == [1.0, -1.0]
    expected = {"target_kl": 1.25, "target_stat": "kl_rms", "bracket": (0.001, 256.0), "device": "cpu", "T": 20, "do_sample": True, "seed": 0}
    assert all({key: value for key, value in call.items() if key != "sign"} == expected for call in calls)
    assert [item["magnitude"] for item in record["signed_predictions"]] == [3.0, 3.0]
    assert record["signed_predictions"][1]["search_history"][-1]["final"] is True
    assert "boundary" not in record
    planned_observations = [
        {"case_id": record["case"]["case_id"], "target_id": record["target_id"], "magnitude": dose["magnitude"], "side": dose["side"], "useful": True, "coherent": True, "provenance": "transfer-judge", "generation_health": {}}
        for dose in final_dose_plan(record)["coefficients"]
    ]
    classified = classify_transfer_boundary(record, planned_observations)
    assert classified["signed_boundaries"] == {"+C": "measured_generation_health_boundary", "-C": "measured_generation_health_boundary"}
    with pytest.raises(ValueError, match="nearby observed"):
        classify_transfer_boundary(record, [])
    for changed, message in (
        ([{**row, "case_id": "calibration"} for row in planned_observations], "case and target"),
        ([{**row, "case_id": "other-case"} for row in planned_observations], "case and target"),
        ([{key: value for key, value in row.items() if key != "target_id"} for row in planned_observations], "case_id, target_id, magnitude and side"),
        ([{**row, "target_id": "other-target"} for row in planned_observations], "case and target"),
        (planned_observations[:2], "planned.*doses"),
        (planned_observations + [{**planned_observations[0], "magnitude": 0.9}], "planned.*doses"),
    ):
        with pytest.raises(ValueError, match=message):
            classify_transfer_boundary(record, changed)


def test_transfer_boundaries_are_classified_per_side():
    prediction = {
        "target_id": "target-a", "target_rms": 1.0, "bracket": [0.001, 256.0],
        "method": "vjp_cache", "case": {"case_id": "transfer-a", "dataset": "synthetic", "prompt_ids": ["SYN-001"]},
        "signed_predictions": [
            {"side": "+C", "magnitude": 256.0, "search_history": [{"coeff": 256.0, "kl_rms": 0.5}]},
            {"side": "-C", "magnitude": 1.0, "search_history": [{"coeff": -1.0, "kl_rms": 1.0}]},
        ],
    }
    observations = [
        {"case_id": "transfer-a", "target_id": "target-a", "magnitude": dose["magnitude"], "side": dose["side"], "generation_health": {"reasons": []}}
        for dose in final_dose_plan(prediction)["coefficients"]
    ]
    classified = classify_transfer_boundary(prediction, observations)
    assert classified["signed_boundaries"] == {"+C": "search_limit", "-C": "measured_generation_health_boundary"}


def test_final_dose_plan_uses_predicted_plus_fixed_nearby_doses():
    prediction = {
        "target_id": "target-a",
        "method": "vjp_cache",
        "case": {"case_id": "transfer-a", "dataset": "synthetic", "prompt_ids": ["SYN-001"]},
        "signed_predictions": [
            {"side": "+C", "magnitude": 0.5, "search_history": []},
            {"side": "-C", "magnitude": 0.6, "search_history": []},
        ],
    }
    plan = final_dose_plan(prediction)
    assert plan["coefficients"] == [
        {"side": "+C", "magnitude": 0.4}, {"side": "+C", "magnitude": 0.5}, {"side": "+C", "magnitude": 0.6},
        {"side": "-C", "magnitude": 0.48}, {"side": "-C", "magnitude": 0.6}, {"side": "-C", "magnitude": 0.72},
    ]
    with pytest.raises(ValueError, match="prediction for each side"):
        final_dose_plan(prediction | {"signed_predictions": prediction["signed_predictions"][:1]})


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
