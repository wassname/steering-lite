"""Reserved production-cache stages for the later Modal sweep."""
from __future__ import annotations

from pathlib import Path

from .cache import cached_stage, reserve, settle
from .validation import numbered_requests, validate_persona_examples


def production_stage(root: Path, ledger: Path, *, stage: str, model: dict, data: dict, method: str, config: dict, prompts: list[str], backend) -> dict:
    """Dispatch only on a production-cache miss after reserving its upper cost."""
    dispatched = False

    def compute() -> dict:
        nonlocal dispatched
        reservation = reserve(ledger, f"modal-{stage}-{method}", config["upper_usd"], limit_usd=50.0)
        dispatched = True
        result = backend.gpu(stage=stage, method=method, config=config, prompts=prompts)
        settle(ledger, reservation, result["actual_usd"])
        return result | {"reservation": reservation}

    result = cached_stage(root / "cache", stage, model=model, data=data, method=method, config=config, prompts=prompts, compute=compute)
    return result | {"reused": not dispatched}


def record_completed_stage(root: Path, *, stage: str, model: dict, data: dict, method: str, config: dict, prompts: list[str], result: dict) -> dict:
    """Put a completed external stage into the production cache without dispatching it again."""
    return cached_stage(
        root / "cache", stage, model=model, data=data, method=method, config=config,
        prompts=prompts, compute=lambda: result,
    )


def persist_local_judge_work(root: Path, *, model: dict, data: dict, rows: list[dict], persona_examples: list[dict], endpoint: str) -> dict:
    """Validate persona pairs and persist unsent target-aware and blind requests."""
    prompts = [row["prompt"] for row in rows]
    return cached_stage(
        root / "cache",
        "local-judge-work",
        model=model,
        data=data,
        method="judge",
        config={"endpoint": endpoint, "persona_pairs": len(persona_examples)},
        prompts=prompts,
        compute=lambda: {
            "persona_checks": validate_persona_examples(persona_examples),
            "requests": numbered_requests(rows, model["judge_model"], endpoint),
        },
    )


def run_stages(root: Path, ledger: Path, stages: list[dict], backend) -> list[dict]:
    """Run ordered GPU stages; local judge work is intentionally not dispatched here."""
    records = []
    for item in stages:
        records.append(production_stage(root, ledger, backend=backend, **item))
    return records
