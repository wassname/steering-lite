"""Method/model signed RMS-KL target fitting and disjoint transfer predictions."""
from __future__ import annotations

import math
from dataclasses import dataclass

from steering_lite.calibrate import calibrate_iso_kl, measure_kl

from .cache import cached_stage, content_key


@dataclass(frozen=True)
class Case:
    case_id: str
    dataset: str
    prompt_ids: tuple[str, ...]


CALIBRATION_CASE = Case("bsbench-v2-calibration", "bsbench-v2-dev", ("BSV2-001", "BSV2-002", "BSV2-003", "BSV2-004"))
EVALUATION_CASE = Case("bsbench-v2-evaluation", "bsbench-v2-dev", tuple(f"BSV2-{number:03d}" for number in range(1, 21)))
TRANSFER_CASES = (
    Case("bsbench-v2-heldout-a", "bsbench-v2-heldout", ("BSV2-021", "BSV2-022")),
    Case("bsbench-v2-heldout-b", "bsbench-v2-heldout", ("BSV2-023", "BSV2-024")),
    Case("paper-native-false-claim-agreement-a", "paper-native-false-claim-agreement", ("PNFCA-001", "PNFCA-002")),
    Case("paper-native-false-claim-agreement-b", "paper-native-false-claim-agreement", ("PNFCA-003", "PNFCA-004")),
)

SIDES = ("+C", "-C")
FINAL_DOSE_MULTIPLIERS = (0.8, 1.0, 1.2)
BENCHMARK_KL_SPEC = {
    "T": 20,
    "do_sample": True,
    "seed": 0,
    "target_stat": "kl_rms",
    "bracket": [0.001, 256.0],
}
PREDICTION_CASES = (EVALUATION_CASE, *TRANSFER_CASES)
_HEALTH_FIELDS = ("rep", "gen_len", "steer_tail", "per_t_mean", "per_t_p90", "per_t_p95", "per_t_max", "per_t_n")


def validate_cases(calibration: Case, transfer: tuple[Case, ...] | list[Case]) -> None:
    cases = (calibration, *transfer)
    if len(cases) < 2 or len({case.case_id for case in cases}) != len(cases):
        raise ValueError("calibration and transfer case IDs must be unique")
    seen: set[str] = set()
    for case in cases:
        if overlap := seen.intersection(case.prompt_ids):
            raise ValueError(f"calibration and transfer cases overlap: {sorted(overlap)}")
        seen.update(case.prompt_ids)


def health_evidence(metrics: dict) -> dict:
    return {field: metrics[field] for field in _HEALTH_FIELDS}


def _signed_coefficient(magnitude: float, side: str) -> float:
    if side not in SIDES or not math.isfinite(magnitude) or magnitude <= 0:
        raise ValueError("a signed dose requires a finite positive magnitude and +C/-C side")
    return magnitude if side == "+C" else -magnitude


def highest_healthy_candidate(observed: list[dict]) -> dict:
    """Choose the largest magnitude whose generation health is clean on both sides."""
    by_magnitude: dict[float, dict[str, dict]] = {}
    for row in observed:
        magnitude = float(row["magnitude"])
        side = row["side"]
        if side not in SIDES or not math.isfinite(magnitude) or magnitude <= 0:
            raise ValueError("candidate observations require positive magnitudes and +C/-C sides")
        if side in by_magnitude.setdefault(magnitude, {}):
            raise ValueError("candidate observations must contain each magnitude/side once")
        by_magnitude[magnitude][side] = row
    healthy = [
        magnitude for magnitude, sides in by_magnitude.items()
        if set(sides) == set(SIDES)
        and all(isinstance(row.get("generation_health"), dict) and row["generation_health"].get("reasons") == [] for row in sides.values())
    ]
    if not healthy:
        raise ValueError("calibration needs one magnitude with clean generation health on both sides")
    magnitude = max(healthy)
    for row in by_magnitude[magnitude].values():
        row["provenance"]
    return {"magnitude": magnitude, "sides": by_magnitude[magnitude]}


def classify_boundary(history: list[dict], observed: list[dict], target_rms: float, bracket: tuple[float, float]) -> str:
    if any(row.get("generation_health", {}).get("reasons") for row in observed):
        return "measured_generation_health_failure"
    if history and max(abs(float(row["coeff"])) for row in history) >= bracket[1] and all(float(row["kl_rms"]) < target_rms for row in history):
        return "search_limit"
    return "measured_generation_health_boundary"


def _runtime_kwargs(kl_spec: dict, runtime_kwargs: dict) -> dict:
    statistical = {"T", "do_sample", "seed", "target_stat", "bracket", "sign", "target_kl"}
    if statistical.intersection(runtime_kwargs):
        raise ValueError("runtime KL kwargs cannot override the persisted statistical specification")
    if kl_spec != BENCHMARK_KL_SPEC:
        raise ValueError("KL calibration requires the benchmark statistical specification")
    return runtime_kwargs | {key: kl_spec[key] for key in ("T", "do_sample", "seed")}


def fit_target(vector, model, tokenizer, prompts, calibration: Case, observed: list[dict], *, method: str, model_id: str, kl_spec: dict, measure_kwargs: dict, measure=measure_kl) -> dict:
    """Pool the two signed RMS-KL measurements at the largest jointly healthy magnitude."""
    boundary = highest_healthy_candidate(observed)
    call_kwargs = _runtime_kwargs(kl_spec, measure_kwargs)
    components = []
    for side in SIDES:
        vector.cfg.coeff = _signed_coefficient(boundary["magnitude"], side)
        metrics = measure(vector, model, tokenizer, prompts, **call_kwargs)
        rms = float(metrics["kl_rms"])
        n_pos = int(metrics["n_pos"])
        if not math.isfinite(rms) or rms < 0 or n_pos <= 0:
            raise ValueError("signed RMS-KL measurement requires finite RMS and positive token count")
        components.append({"side": side, "magnitude": boundary["magnitude"], "kl_rms": rms, "n_pos": n_pos, "health": health_evidence(metrics)})
    total_n = sum(component["n_pos"] for component in components)
    target_rms = math.sqrt(sum(component["n_pos"] * component["kl_rms"] ** 2 for component in components) / total_n)
    source = {"method": method, "model": model_id, "case": {"case_id": calibration.case_id, "dataset": calibration.dataset, "prompt_ids": list(calibration.prompt_ids)}, "magnitude": boundary["magnitude"], "provenance": {side: boundary["sides"][side]["provenance"] for side in SIDES}, "kl_spec": kl_spec}
    return {
        "schema": "bsbench-signed-rms-kl-target-v2",
        "target_id": content_key(source | {"target_rms": target_rms, "components": components}),
        "target_stat": kl_spec["target_stat"],
        "target_rms": target_rms,
        "kl_spec": kl_spec,
        "source": source,
        "signed_components": components,
        "observed_boundary": boundary,
    }


def predict_transfer(vector, model, tokenizer, prompts, target: dict, case: Case, *, kl_spec: dict, solver_kwargs: dict, solver=calibrate_iso_kl) -> dict:
    """Solve positive and negative magnitudes separately against one pooled RMS target."""
    if target["target_stat"] != "kl_rms" or target.get("kl_spec") != kl_spec:
        raise ValueError("transfer target and solver require the same RMS-KL specification")
    call_kwargs = _runtime_kwargs(kl_spec, solver_kwargs)
    bracket = tuple(kl_spec["bracket"])
    signed_predictions = []
    for side in SIDES:
        sign = 1.0 if side == "+C" else -1.0
        coefficient, history = solver(vector, model, tokenizer, prompts, target_kl=target["target_rms"], target_stat=kl_spec["target_stat"], bracket=bracket, sign=sign, **call_kwargs)
        coefficient = float(coefficient)
        magnitude = abs(coefficient)
        if not math.isfinite(magnitude) or magnitude == 0 or (coefficient > 0) != (sign > 0):
            raise ValueError("signed transfer solver returned an invalid coefficient")
        signed_predictions.append({"side": side, "magnitude": magnitude, "search_history": history})
    return {
        "schema": "bsbench-signed-rms-kl-transfer-v2",
        "target_id": target["target_id"],
        "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)},
        "method": target["source"]["method"],
        "model": target["source"]["model"],
        "target_stat": kl_spec["target_stat"],
        "target_rms": target["target_rms"],
        "bracket": kl_spec["bracket"],
        "kl_spec": kl_spec,
        "signed_predictions": signed_predictions,
    }


def classify_transfer_boundary(prediction: dict, nearby_observed: list[dict]) -> dict:
    if not nearby_observed:
        raise ValueError("transfer boundary classification needs nearby observed doses")
    plan = final_dose_plan(prediction)
    required = {"case_id", "target_id", "magnitude", "side"}
    if any(not required.issubset(row) for row in nearby_observed):
        raise ValueError("transfer observations require case_id, target_id, magnitude and side")
    if any(row["case_id"] != plan["case"]["case_id"] or row["target_id"] != plan["target_id"] for row in nearby_observed):
        raise ValueError("transfer observations must match the predicted case and target")
    actual = {(row["side"], float(row["magnitude"])) for row in nearby_observed}
    expected = {(dose["side"], float(dose["magnitude"])) for dose in plan["coefficients"]}
    if actual != expected or len(nearby_observed) != len(expected):
        raise ValueError("transfer observations must cover exactly the planned signed doses")
    signed_boundaries = {
        side: classify_boundary(
            next(item["search_history"] for item in prediction["signed_predictions"] if item["side"] == side),
            [row for row in nearby_observed if row["side"] == side],
            prediction["target_rms"],
            tuple(prediction["bracket"]),
        )
        for side in SIDES
    }
    return prediction | {"signed_boundaries": signed_boundaries, "nearby_observed": nearby_observed}


def final_dose_plan(prediction: dict) -> dict:
    """Generate 0.8/1.0/1.2 times each separately solved signed magnitude."""
    signed = prediction.get("signed_predictions")
    if not isinstance(signed, list) or {item.get("side") for item in signed} != set(SIDES) or len(signed) != 2:
        raise ValueError("final dose plan requires one prediction for each side")
    doses = []
    for side in SIDES:
        item = next(item for item in signed if item["side"] == side)
        magnitude = float(item["magnitude"])
        _signed_coefficient(magnitude, side)
        side_doses = [round(magnitude * multiplier, 12) for multiplier in FINAL_DOSE_MULTIPLIERS]
        if len(set(side_doses)) != len(side_doses):
            raise ValueError("final dose plan requires distinct predicted and nearby doses")
        doses.extend({"side": side, "magnitude": dose, "multiplier": multiplier} for dose, multiplier in zip(side_doses, FINAL_DOSE_MULTIPLIERS, strict=True))
    return {"schema": "bsbench-signed-final-dose-plan-v2", "case": prediction["case"], "target_id": prediction["target_id"], "method": prediction["method"], "signed_predictions": signed, "coefficients": doses}


def cached_search(root, *, model, data, method, config, prompts, compute):
    return cached_stage(root, "rms-kl-search", model=model, data=data, method=method, config=config, prompts=prompts, compute=compute)
