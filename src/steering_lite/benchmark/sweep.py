"""Dry BS-bench sweep manifest with cached stages and auditable spending bounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .cache import cached_stage, committed, reserve_many, save_json
from .dose_search import CALIBRATION_CASE, TRANSFER_CASES
from .generation import cohort_identity, read_dev_cohort
from .pipeline import METHODS

MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL = "deepseek/deepseek-chat"
BUDGET_LIMIT_USD = 50.0
PERSONA_VALIDATION_PAIRS = 12

MODAL_SOURCE = "https://modal.com/pricing (accessed 2026-09-19)"
JUDGE_SOURCE = "https://openrouter.ai/deepseek/deepseek-chat/overview?tab=parameters (accessed 2026-09-19)"
MODAL_A10G_USD_PER_GPU_HOUR = 1.10
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
JUDGE_INPUT_USD_PER_MTOKEN = 0.2574
JUDGE_OUTPUT_USD_PER_MTOKEN = 1.029
GPU_HOURS_PER_STAGE = 0.5
CPU_CORES_PER_GPU_STAGE = 1.0
MEMORY_GIB_PER_GPU_STAGE = 4.0


@dataclass(frozen=True)
class BudgetLine:
    kind: str
    quantity: float
    unit: str
    rate_usd: float
    rate_unit: str
    rate_source: str

    @property
    def subtotal_usd(self) -> float:
        return self.quantity * self.rate_usd

    def record(self) -> dict:
        return asdict(self) | {"subtotal_usd": self.subtotal_usd}


def _sum(lines: list[BudgetLine]) -> float:
    return sum(line.subtotal_usd for line in lines)


def _request_counts(stages: list[dict], questions: int) -> dict[str, int]:
    target_aware_stages = sum(stage["stage"] == "target-aware-requests" for stage in stages)
    blind_stages = sum(stage["stage"] == "blind-requests" for stage in stages)
    return {
        "target_aware": questions * target_aware_stages * 2,
        "blind": questions * blind_stages * 2,
        "persona_validation": PERSONA_VALIDATION_PAIRS,
    }


def cost_estimate(stages: list[dict], questions: int) -> dict:
    gpu_stages = sum(stage["runner"] == "modal_gpu" for stage in stages)
    gpu_hours = gpu_stages * GPU_HOURS_PER_STAGE
    request_counts = _request_counts(stages, questions)
    input_tokens = (
        request_counts["target_aware"] * 4_000
        + request_counts["blind"] * 2_000
        + request_counts["persona_validation"] * 2_000
    )
    output_tokens = (
        (request_counts["target_aware"] + request_counts["blind"]) * 1_200
        + request_counts["persona_validation"] * 100
    )
    expected = [
        BudgetLine("Modal A10G GPU", gpu_hours, "GPU-hour", MODAL_A10G_USD_PER_GPU_HOUR, "USD/GPU-hour", MODAL_SOURCE),
        BudgetLine("Modal CPU", gpu_hours * 3600 * CPU_CORES_PER_GPU_STAGE, "core-second", MODAL_CPU_USD_PER_CORE_SECOND, "USD/core-second", MODAL_SOURCE),
        BudgetLine("Modal memory", gpu_hours * 3600 * MEMORY_GIB_PER_GPU_STAGE, "GiB-second", MODAL_MEMORY_USD_PER_GIB_SECOND, "USD/GiB-second", MODAL_SOURCE),
        BudgetLine("Judge input", input_tokens / 1_000_000, "M-token", JUDGE_INPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
        BudgetLine("Judge output", output_tokens / 1_000_000, "M-token", JUDGE_OUTPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
    ]
    unresolved = [
        BudgetLine("Unresolved Modal A10G", 4.0, "GPU-hour", MODAL_A10G_USD_PER_GPU_HOUR, "USD/GPU-hour", MODAL_SOURCE),
        BudgetLine("Unresolved Modal CPU", 4.0 * 3600, "core-second", MODAL_CPU_USD_PER_CORE_SECOND, "USD/core-second", MODAL_SOURCE),
        BudgetLine("Unresolved Modal memory", 4.0 * 3600 * MEMORY_GIB_PER_GPU_STAGE, "GiB-second", MODAL_MEMORY_USD_PER_GIB_SECOND, "USD/GiB-second", MODAL_SOURCE),
        BudgetLine("Unresolved judge input", 100 * 4_000 / 1_000_000, "M-token", JUDGE_INPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
        BudgetLine("Unresolved judge output", 100 * 1_200 / 1_000_000, "M-token", JUDGE_OUTPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
    ]
    expected_usd = _sum(expected)
    retry_usd = expected_usd
    unresolved_usd = _sum(unresolved)
    reservations = [
        {"kind": "expected_modal_and_judge_work", "upper_usd": expected_usd},
        {"kind": "one_full_retry", "upper_usd": retry_usd},
        {"kind": "unresolved_work", "upper_usd": unresolved_usd},
    ]
    return {
        "schema": "bsbench-budget-estimate-v2",
        "judge_model": JUDGE_MODEL,
        "planning_assumptions": {"gpu_hours_per_stage_upper": GPU_HOURS_PER_STAGE, "cpu_cores_per_gpu_stage": CPU_CORES_PER_GPU_STAGE, "memory_gib_per_gpu_stage": MEMORY_GIB_PER_GPU_STAGE, "target_aware_input_tokens_per_request": 4_000, "blind_input_tokens_per_request": 2_000, "output_tokens_per_request": 1_200},
        "quantities": {"gpu_stages": gpu_stages, "gpu_hours": gpu_hours, "requests": request_counts, "input_tokens": input_tokens, "output_tokens": output_tokens},
        "expected_work": [line.record() for line in expected],
        "retry_reserve": {"copies_of_expected_work": 1, "subtotal_usd": retry_usd},
        "unresolved_reserve": [line.record() for line in unresolved],
        "planned_reservations": reservations,
        "total_upper_usd": expected_usd + retry_usd + unresolved_usd,
    }


def preflight_budget(ledger: Path, estimate: dict) -> dict:
    existing = committed(ledger)
    total = existing + estimate["total_upper_usd"]
    if total >= BUDGET_LIMIT_USD:
        raise RuntimeError(f"budget: ${total:.2f} is at or above ${BUDGET_LIMIT_USD:.2f}")
    return estimate | {"limit_usd": BUDGET_LIMIT_USD, "existing_committed_usd": existing, "total_upper_usd": total}


def reserve_budget(ledger: Path, estimate: dict) -> list[str]:
    preflight_budget(ledger, estimate)
    return reserve_many(
        ledger,
        [(item["kind"], item["upper_usd"]) for item in estimate["planned_reservations"]],
        limit_usd=BUDGET_LIMIT_USD,
        strict_limit=True,
    )


def cached_dry_stage(root: Path, *, stage: str, runner: str, model: dict, data: dict, method: str, config: dict, prompts: list[str], code: str | None = None) -> dict:
    computed = False

    def compute() -> dict:
        nonlocal computed
        computed = True
        return {"schema": "bsbench-dry-stage-v1", "stage": stage, "runner": runner}

    result = cached_stage(root, stage, model=model, data=data, method=method, config=config, prompts=prompts, code=code, compute=compute)
    return result | {"reused": not computed}


def case_identity(case) -> dict:
    return {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)}


def condition_stages(method: str) -> tuple[tuple[str, str], ...]:
    common = (("generation", "modal_gpu"), ("generation-health", "local"), ("target-aware-requests", "local_judge_api"), ("blind-requests", "local_judge_api"))
    if method in {"bare", "prompting"}:
        return common
    return (("extract", "modal_gpu"), ("rms-kl-fit", "modal_gpu"), ("rms-kl-transfer", "modal_gpu"), *common)


def dry_manifest(out: Path, model_id: str = MODEL_ID) -> dict:
    rows = read_dev_cohort()
    prompts = [row["prompt"] for row in rows]
    data = cohort_identity(rows)
    model = {"id": model_id}
    cache_root = out / "dry-plan-cache"
    ledger = out / "costs.jsonl"
    stages = []
    for method in METHODS:
        config = {"condition": method, "target_stat": "kl_rms" if method not in {"bare", "prompting"} else None, "calibration_case": case_identity(CALIBRATION_CASE) if method not in {"bare", "prompting"} else None, "transfer_cases": [case_identity(case) for case in TRANSFER_CASES] if method not in {"bare", "prompting"} else []}
        for stage, runner in condition_stages(method):
            stages.append({"method": method, **cached_dry_stage(cache_root, stage=stage, runner=runner, model=model, data=data, method=method, config=config | {"stage": stage}, prompts=prompts)})
    manifest = {
        "schema": "bsbench-sweep-manifest-v1", "mode": "dry-run", "paid_execution_enabled": False,
        "model": model, "questions": [{"question_id": row["question_id"], "question_number": row["question_number"]} for row in rows],
        "data": data, "conditions": list(METHODS), "judge_model": JUDGE_MODEL, "target_aware_request_schema": "bsbench-judge-request-v1", "blind_request_schema": "bsbench-judge-request-v1",
        "ledger": str(ledger), "production_cache": str(out / "cache"), "dry_plan_cache": str(cache_root), "stages": stages,
    }
    manifest["cost_estimate"] = preflight_budget(ledger, cost_estimate(stages, len(rows)))
    save_json(out / "manifest.json", manifest)
    save_json(out / "cost-estimate.json", manifest["cost_estimate"])
    return manifest
