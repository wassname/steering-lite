"""Dry BS-bench sweep manifest with cached stages and auditable spending bounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

from .cache import cached_stage, committed, content_key, reserve_many, save_json
from .dose_search import (
    CALIBRATION_CASE,
    FINAL_DOSE_MULTIPLIERS,
    TRANSFER_CASES,
    Case,
    final_dose_plan,
    validate_cases,
)
from .generation import cohort_identity, read_dev_cohort
from .pipeline import METHODS
from .transfer_data import PromptRecord, load_transfer_records, transfer_provenance, transfer_records_identity

MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL = "deepseek/deepseek-chat"
BUDGET_LIMIT_USD = 50.0
PERSONA_VALIDATION_PAIRS = 12
# Conservative behavioral-candidate policy cap for Phase A; independent of solver iterations.
CANDIDATE_DOSE_UPPER = 12
BSBENCH_PERSONAS = (("sycophantic", "abrasive"),)
BSBENCH_PERSONA_TEMPLATE = "Answer as someone who is {persona}."
BSBENCH_PERSONA_SEED = 0
BSBENCH_PERSONA_N_PAIRS = 256
BSBENCH_PERSONA_THINKING = True

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
MODAL_GPU_STAGE_UPPER_USD = GPU_HOURS_PER_STAGE * (
    MODAL_A10G_USD_PER_GPU_HOUR
    + 3600 * CPU_CORES_PER_GPU_STAGE * MODAL_CPU_USD_PER_CORE_SECOND
    + 3600 * MEMORY_GIB_PER_GPU_STAGE * MODAL_MEMORY_USD_PER_GIB_SECOND
)


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


def _request_counts(stages: list[dict]) -> dict[str, int]:
    target_aware = sum(
        2 * stage["item_count"]
        for stage in stages
        if stage["stage"] in {"target-aware-requests", "candidate-aware", "final-aware"}
    )
    blind = sum(
        2 * stage["item_count"]
        for stage in stages
        if stage["stage"] in {"blind-requests", "candidate-blind", "final-blind"}
    )
    return {
        "target_aware": target_aware,
        "blind": blind,
        "persona_validation": PERSONA_VALIDATION_PAIRS,
    }


def cost_estimate(stages: list[dict]) -> dict:
    gpu_stages = sum(stage["runner"] == "modal_gpu" for stage in stages)
    gpu_hours = gpu_stages * GPU_HOURS_PER_STAGE
    request_counts = _request_counts(stages)
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
        "planning_assumptions": {"gpu_hours_per_stage_upper": GPU_HOURS_PER_STAGE, "cpu_cores_per_gpu_stage": CPU_CORES_PER_GPU_STAGE, "memory_gib_per_gpu_stage": MEMORY_GIB_PER_GPU_STAGE, "modal_gpu_stage_upper_usd": MODAL_GPU_STAGE_UPPER_USD, "target_aware_input_tokens_per_request": 4_000, "blind_input_tokens_per_request": 2_000, "output_tokens_per_request": 1_200},
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


def phase_b_budget_stages() -> tuple[dict, ...]:
    """Reserve the offline-described vector-method final sweep; this never dispatches."""
    records = load_transfer_records()
    item_count = sum(len(records[case.case_id]) for case in TRANSFER_CASES) * len(FINAL_DOSE_MULTIPLIERS)
    provenance = transfer_records_identity(records)
    return tuple(
        {
            "stage": stage,
            "runner": runner,
            "method": method,
            "item_count": item_count,
            "transfer_provenance_sha256": content_key(provenance),
        }
        for method in METHODS
        if method not in {"bare", "prompting"}
        for stage, runner in (
            ("final-generation", "modal_gpu"),
            ("final-health", "local"),
            ("final-aware", "local_judge_api"),
            ("final-blind", "local_judge_api"),
        )
    )


def final_stages(
    *,
    method: str,
    vector_sha256: str,
    observed: list[dict],
    transfer_predictions: list[dict],
    case_prompts: dict[str, tuple[PromptRecord, ...] | list[PromptRecord]],
    prompt_spec: dict,
    transfer_cases: tuple[Case, ...] = TRANSFER_CASES,
) -> tuple[dict, ...]:
    """Describe the post-judgment generation graph without dispatching it."""
    if not vector_sha256 or not observed or not transfer_predictions or not case_prompts or not prompt_spec:
        raise ValueError("final stages require vector, observed records, transfer predictions, prompts and prompt spec")
    validate_cases(CALIBRATION_CASE, transfer_cases)
    case_ids = [case.case_id for case in transfer_cases]
    if set(case_prompts) != set(case_ids):
        raise ValueError("final stages require complete prompt records for every transfer case")
    actual_case_records = {case_id: tuple(case_prompts[case_id]) for case_id in case_ids}
    for case in transfer_cases:
        records = actual_case_records[case.case_id]
        if not records or any(not isinstance(record, PromptRecord) for record in records):
            raise ValueError("final stages require auditable PromptRecord transfer data")
        if tuple(record.prompt_id for record in records) != case.prompt_ids:
            raise ValueError("final stages require complete prompt records matching each transfer case")
        if any(record.dataset != case.dataset for record in records):
            raise ValueError("final stages require prompt-record dataset provenance matching each transfer case")
        if any(
            not record.prompt or not record.source_path or not record.source_revision or not record.source_sha256
            or not record.answer_key
            or record.content_sha256 != hashlib.sha256(record.prompt.encode()).hexdigest()
            or record.answer_key_sha256 != hashlib.sha256(record.answer_key.encode()).hexdigest()
            for record in records
        ):
            raise ValueError("final stages require loadable non-placeholder prompt records with valid provenance")
    actual_case_prompts = {
        case_id: [record.prompt for record in records]
        for case_id, records in actual_case_records.items()
    }

    expected_cases = {case.case_id: case_identity(case) for case in transfer_cases}
    target_ids = set()
    for prediction in transfer_predictions:
        if not isinstance(prediction, dict):
            raise ValueError("final stages require transfer prediction records")
        if prediction.get("schema") != "bsbench-rms-kl-transfer-v1":
            raise ValueError("final stages require RMS-KL transfer prediction records")
        if prediction.get("method") != method:
            raise ValueError("final stages require transfer predictions for the requested method")
        case = prediction.get("case")
        if not isinstance(case, dict) or case.get("case_id") not in expected_cases or case != expected_cases[case["case_id"]]:
            raise ValueError("final stages require complete matching transfer case identities")
        target_id = prediction.get("target_id")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("final stages require a transfer target ID")
        target_ids.add(target_id)
        try:
            coefficient = float(prediction["predicted_coefficient"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("final stages require a numeric predicted coefficient") from error
        if not math.isfinite(coefficient) or coefficient == 0.0:
            raise ValueError("final stages require a finite non-zero predicted coefficient")
    if len(target_ids) != 1:
        raise ValueError("final stages require one common transfer target ID")
    ordered_observed = sorted(observed, key=content_key)
    plans = [final_dose_plan(prediction) for prediction in transfer_predictions]
    plans_by_case = {plan["case"]["case_id"]: plan for plan in plans}
    if set(plans_by_case) != set(case_ids) or len(plans_by_case) != len(plans):
        raise ValueError("final stages require one transfer prediction for every transfer case")
    ordered_plans = [plans_by_case[case_id] for case_id in case_ids]
    prompt_hashes = {
        case_id: content_key({"prompts": prompts})
        for case_id, prompts in actual_case_prompts.items()
    }
    item_count = sum(
        len(actual_case_prompts[plan["case"]["case_id"]]) * len(plan["coefficients"])
        for plan in ordered_plans
    )
    config = {
        "schema": "bsbench-final-stage-v1",
        "method": method,
        "vector_sha256": vector_sha256,
        "observed_sha256": content_key({"observed": ordered_observed}),
        "candidate_coefficients": sorted(row["coefficient"] for row in ordered_observed),
        "final_dose_plans": ordered_plans,
        "final_dose_plans_sha256": content_key({"plans": ordered_plans}),
        "case_prompt_hashes": prompt_hashes,
        "case_prompt_provenance": {
            case_id: transfer_provenance(records)
            for case_id, records in actual_case_records.items()
        },
        "prompt_spec": prompt_spec,
        "prompt_spec_sha256": content_key(prompt_spec),
    }
    generation = {
        "stage": "final-generation",
        "runner": "modal_gpu",
        "method": method,
        "config": config,
        "item_count": item_count,
        "case_prompts": actual_case_prompts,
        "case_prompt_records": {
            case_id: transfer_provenance(records)
            for case_id, records in actual_case_records.items()
        },
        "generation_plan": [
            {
                "case": plan["case"],
                "prompts": actual_case_prompts[plan["case"]["case_id"]],
                "coefficients": plan["coefficients"],
            }
            for plan in ordered_plans
        ],
    }
    downstream = tuple(
        {
            "stage": stage,
            "runner": runner,
            "method": method,
            "config": config,
            "item_count": item_count,
            "input_stage": "final-generation",
        }
        for stage, runner in (
            ("final-health", "local"),
            ("final-aware", "local_judge_api"),
            ("final-blind", "local_judge_api"),
        )
    )
    return (generation, *downstream)


def condition_stages(method: str) -> tuple[tuple[str, str], ...]:
    if method in {"bare", "prompting"}:
        return (("generation", "modal_gpu"), ("generation-health", "local"), ("target-aware-requests", "local_judge_api"), ("blind-requests", "local_judge_api"))
    return (("calibration-candidates", "modal_gpu"), ("candidate-health", "local"), ("candidate-aware", "local_judge_api"), ("candidate-blind", "local_judge_api"))


def dry_manifest(out: Path, model_id: str = MODEL_ID) -> dict:
    rows = read_dev_cohort()
    prompts = [row["prompt"] for row in rows]
    prompts_by_id = {row["question_id"]: row["prompt"] for row in rows}
    calibration_prompts = [prompts_by_id[prompt_id] for prompt_id in CALIBRATION_CASE.prompt_ids]
    data = cohort_identity(rows)
    model = {"id": model_id}
    cache_root = out / "dry-plan-cache"
    ledger = out / "costs.jsonl"
    stages = []
    persona_source = {
        "pairs": [list(pair) for pair in BSBENCH_PERSONAS],
        "template": BSBENCH_PERSONA_TEMPLATE,
        "seed": BSBENCH_PERSONA_SEED,
        "n_pairs": BSBENCH_PERSONA_N_PAIRS,
        "thinking": BSBENCH_PERSONA_THINKING,
    }
    for method in METHODS:
        vector_method = method not in {"bare", "prompting"}
        stage_prompts = calibration_prompts if vector_method else prompts
        config = {
            "condition": method,
            "target_stat": "kl_rms" if vector_method else None,
            "calibration_case": case_identity(CALIBRATION_CASE) if vector_method else None,
            "calibration_prompts_sha256": content_key({"prompts": calibration_prompts}) if vector_method else None,
            "persona_source": persona_source if vector_method else None,
            "persona_source_sha256": content_key(persona_source) if vector_method else None,
            "candidate_dose_upper": CANDIDATE_DOSE_UPPER if vector_method else None,
        }
        for stage, runner in condition_stages(method):
            stage_config = config | {"stage": stage}
            if runner == "modal_gpu":
                stage_config |= {"upper_usd": MODAL_GPU_STAGE_UPPER_USD}
            stages.append(
                {
                    "method": method,
                    "item_count": len(calibration_prompts) * CANDIDATE_DOSE_UPPER if vector_method else len(stage_prompts),
                    "prompts_sha256": content_key({"prompts": stage_prompts}),
                    "config": stage_config,
                    **cached_dry_stage(
                        cache_root,
                        stage=stage,
                        runner=runner,
                        model=model,
                        data=data,
                        method=method,
                        config=stage_config,
                        prompts=stage_prompts,
                    ),
                }
            )
    final_budget_stages = phase_b_budget_stages()
    manifest = {
        "schema": "bsbench-sweep-manifest-v1", "mode": "dry-run", "paid_execution_enabled": False,
        "model": model, "questions": [{"question_id": row["question_id"], "question_number": row["question_number"]} for row in rows],
        "data": data, "conditions": list(METHODS), "judge_model": JUDGE_MODEL, "target_aware_request_schema": "bsbench-judge-request-v1", "blind_request_schema": "bsbench-judge-request-v1",
        "ledger": str(ledger), "production_cache": str(out / "cache"), "dry_plan_cache": str(cache_root),
        "stages": stages, "phase_b_budget_stages": list(final_budget_stages),
    }
    manifest["cost_estimate"] = preflight_budget(ledger, cost_estimate(stages + list(final_budget_stages)))
    save_json(out / "manifest.json", manifest)
    save_json(out / "cost-estimate.json", manifest["cost_estimate"])
    return manifest
