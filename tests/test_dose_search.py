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
        "useful": useful,
        "coherent": coherent,
        "provenance": provenance,
        "generation_health": {"rep": 0.0, "gen_len": 40, "steer_tail": "later response"},
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
    assert {case.dataset for case in TRANSFER_CASES if "other-dataset" in case.dataset} == {
        "other-dataset-a-placeholder",
        "other-dataset-b-placeholder",
    }
    with pytest.raises(ValueError, match="overlap"):
        validate_cases(Case("cal", "a", ("one",)), (Case("x", "b", ("two",)), Case("y", "c", ("two",))))


def test_fit_target_measures_observed_useful_coherent_boundary():
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
    assert vector.cfg.coeff == 0.4
    assert calls[0][1] == {"device": "cpu"}
    assert target["target_stat"] == "kl_rms"
    assert target["target_rms"] == 1.25
    assert target["source"]["provenance"] == "judge-002"
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
    classified = classify_transfer_boundary(
        record,
        [observed(0.6, useful=True, coherent=True, provenance="transfer-judge")],
    )
    assert classified["boundary"] == "measured_useful_coherent_boundary"
    with pytest.raises(ValueError, match="nearby observed"):
        classify_transfer_boundary(record, [])


def test_final_dose_plan_uses_predicted_plus_fixed_nearby_doses():
    plan = final_dose_plan({
        "target_id": "target-a",
        "case": {"case_id": "transfer-a"},
        "predicted_coefficient": -0.5,
    })
    assert plan["coefficients"] == [-0.4, -0.5, -0.6]
    with pytest.raises(ValueError, match="non-zero"):
        final_dose_plan({"target_id": "target-a", "case": {"case_id": "transfer-a"}, "predicted_coefficient": 0.0})


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
            observed(0.8, useful=False, coherent=False, provenance="judge-failure"),
        ],
        1.0,
        (0.1, 4.0),
    ) == "measured_coherence_failure"


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
