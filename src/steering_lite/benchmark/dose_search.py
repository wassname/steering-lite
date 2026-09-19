"""Method/model RMS-KL target fitting and disjoint transfer predictions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from steering_lite.calibrate import calibrate_iso_kl, measure_kl

from .cache import cached_stage, content_key


@dataclass(frozen=True)
class Case:
    case_id: str
    dataset: str
    prompt_ids: tuple[str, ...]


CALIBRATION_CASE = Case(
    "bsbench-v2-calibration",
    "bsbench-v2-dev",
    ("BSV2-001", "BSV2-002", "BSV2-003", "BSV2-004"),
)
TRANSFER_CASES = (
    Case("bsbench-v2-dev-transfer", "bsbench-v2-dev", ("BSV2-005", "BSV2-006", "BSV2-007", "BSV2-008")),
    Case("bsbench-v2-heldout-transfer", "bsbench-v2-heldout", ("BSV2-H-001", "BSV2-H-002")),
    Case("other-dataset-a-transfer", "other-dataset-a-placeholder", ("OTHER-A-001", "OTHER-A-002")),
    Case("other-dataset-b-transfer", "other-dataset-b-placeholder", ("OTHER-B-001", "OTHER-B-002")),
)


_HEALTH_FIELDS = (
    "rep",
    "gen_len",
    "steer_tail",
    "per_t_mean",
    "per_t_p90",
    "per_t_p95",
    "per_t_max",
    "per_t_n",
)


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
    """Keep the real calibration and later-generation health measurements."""
    return {field: metrics[field] for field in _HEALTH_FIELDS}


def useful_coherent_boundary(observed: list[dict]) -> dict:
    candidates = [row for row in observed if row["useful"] and row["coherent"]]
    if not candidates:
        raise ValueError("calibration needs a measured useful, coherent dose")
    boundary = max(candidates, key=lambda row: abs(float(row["coefficient"])))
    boundary["provenance"]
    boundary["generation_health"]
    return boundary


def classify_boundary(
    history: list[dict],
    observed: list[dict],
    target_rms: float,
    bracket: tuple[float, float],
) -> str:
    """Classify a measured behavioral failure separately from an RMS search limit."""
    if any(not row["coherent"] for row in observed):
        return "measured_coherence_failure"
    if history and max(abs(float(row["coeff"])) for row in history) >= bracket[1] and all(
        float(row["kl_rms"]) < target_rms for row in history
    ):
        return "search_limit"
    return "measured_useful_coherent_boundary"


def fit_target(
    vector,
    model,
    tokenizer,
    prompts,
    calibration: Case,
    observed: list[dict],
    *,
    method: str,
    model_id: str,
    measure_kwargs: dict,
    measure=measure_kl,
) -> dict:
    """Measure RMS-KL at the observed useful/coherent calibration boundary."""
    boundary = useful_coherent_boundary(observed)
    vector.cfg.coeff = float(boundary["coefficient"])
    metrics = measure(vector, model, tokenizer, prompts, **measure_kwargs)
    target_rms = float(metrics["kl_rms"])
    source = {
        "method": method,
        "model": model_id,
        "case": asdict(calibration),
        "coefficient": vector.cfg.coeff,
        "provenance": boundary["provenance"],
    }
    return {
        "schema": "bsbench-rms-kl-target-v1",
        "target_id": content_key(source | {"target_rms": target_rms}),
        "target_stat": "kl_rms",
        "target_rms": target_rms,
        "source": source,
        "observed_boundary": boundary,
        "calibration_health": health_evidence(metrics),
    }


def predict_transfer(
    vector,
    model,
    tokenizer,
    prompts,
    target: dict,
    case: Case,
    nearby_observed: list[dict],
    *,
    bracket: tuple[float, float],
    solver_kwargs: dict,
    solver=calibrate_iso_kl,
) -> dict:
    """Use the fitted RMS-KL target to solve a coefficient on one new case."""
    if target["target_stat"] != "kl_rms":
        raise ValueError("transfer requires an RMS-KL target")
    if not nearby_observed:
        raise ValueError("transfer needs nearby observed doses")
    coefficient, history = solver(
        vector,
        model,
        tokenizer,
        prompts,
        target_kl=target["target_rms"],
        target_stat="kl_rms",
        bracket=bracket,
        **solver_kwargs,
    )
    return {
        "schema": "bsbench-rms-kl-transfer-v1",
        "target_id": target["target_id"],
        "case": asdict(case),
        "method": target["source"]["method"],
        "model": target["source"]["model"],
        "target_stat": "kl_rms",
        "target_rms": target["target_rms"],
        "predicted_coefficient": coefficient,
        "search_history": history,
        "boundary": classify_boundary(history, nearby_observed, target["target_rms"], bracket),
        "nearby_observed": nearby_observed,
    }


def cached_search(root, *, model, data, method, config, prompts, compute):
    return cached_stage(
        root,
        "rms-kl-search",
        model=model,
        data=data,
        method=method,
        config=config,
        prompts=prompts,
        compute=compute,
    )
