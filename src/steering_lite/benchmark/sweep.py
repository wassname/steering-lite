"""Dry BS-bench sweep manifest with cached stages and auditable spending bounds."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from steering_lite.data import persona_corpus_identity

from .cache import cached_stage, committed, content_key, reserve_many, save_json, valid_cost, source_hash, require_resolved_ledger
from .dose_search import (
    CALIBRATION_CASE,
    EVALUATION_CASE,
    FINAL_DOSE_MULTIPLIERS,
    PREDICTION_CASES,
    TRANSFER_CASES,
    Case,
    final_dose_plan,
    validate_cases,
)
from .generation import cohort_identity, read_dev_cohort
from .pipeline import METHODS
from .transfer_data import PromptRecord, load_evaluation_records, load_transfer_records, transfer_provenance, transfer_records_identity

MODEL_ID = "Qwen/Qwen3.5-4B"
JUDGE_MODEL = "deepseek/deepseek-v4-flash-0731"
BUDGET_LIMIT_USD = 50.0
RANDOM_SEEDS = tuple(range(5))
PERSONA_VALIDATION_PROMPT_IDS = tuple(f"BSV2-{number:03d}" for number in range(1, 13))
PERSONA_VALIDATION_PAIRS = len(PERSONA_VALIDATION_PROMPT_IDS)
# Conservative behavioral-candidate policy cap for Phase A; independent of solver iterations.
CANDIDATE_DOSE_UPPER = 12
BSBENCH_PERSONAS = (("sycophantic", "abrasive"),)
BSBENCH_PERSONA_TEMPLATE = "Answer as someone who is {persona}."
BSBENCH_PERSONA_SEED = 0
BSBENCH_PERSONA_REQUESTED_PAIRS = 256
BSBENCH_PERSONA_ACTUAL_PAIRS = 200
BSBENCH_PERSONA_CORPUS_SHA256 = "9f16c7948e46ab77a2704c6c67529ee9596a2b0ad2614f9faa045695a39319ba"
BSBENCH_PERSONA_THINKING = True

MODAL_SOURCE = "https://modal.com/pricing (accessed 2026-09-19)"
JUDGE_SOURCE = None
MODAL_A10G_USD_PER_GPU_HOUR = 1.10
MODAL_CPU_USD_PER_CORE_SECOND = 0.0000131
MODAL_MEMORY_USD_PER_GIB_SECOND = 0.00000222
JUDGE_INPUT_USD_PER_MTOKEN = None
JUDGE_OUTPUT_USD_PER_MTOKEN = None
MODAL_GPU_STAGE_TIMEOUT_SECONDS = 44 * 60
GPU_HOURS_PER_STAGE = MODAL_GPU_STAGE_TIMEOUT_SECONDS / 3600
CPU_CORES_PER_GPU_STAGE = 1.0
MEMORY_GIB_PER_GPU_STAGE = 4.0
MODAL_GPU_STAGE_UPPER_USD = GPU_HOURS_PER_STAGE * (
    MODAL_A10G_USD_PER_GPU_HOUR
    + 3600 * CPU_CORES_PER_GPU_STAGE * MODAL_CPU_USD_PER_CORE_SECOND
    + 3600 * MEMORY_GIB_PER_GPU_STAGE * MODAL_MEMORY_USD_PER_GIB_SECOND
)


def load_judge_pricing(path: Path) -> dict:
    artifact = json.loads(path.read_text())
    from .judge import REFERENCE_ROUTING
    if artifact["source"] != f"https://openrouter.ai/api/v1/models/{JUDGE_MODEL}/endpoints":
        raise ValueError("pricing requires exact model endpoint metadata, not aggregate minimum rates")
    model = artifact["body"]["data"]
    if model["id"] != JUDGE_MODEL:
        raise ValueError("endpoint metadata must identify the exact V4 Flash model")
    routing = REFERENCE_ROUTING["provider"]
    required = {"temperature", "max_tokens", "min_p", "reasoning", "response_format", "structured_outputs"}
    eligible = [endpoint for endpoint in model["endpoints"] if endpoint["model_id"] == JUDGE_MODEL and endpoint["provider_name"] not in routing["ignore"] and endpoint["quantization"] in routing["quantizations"] and required.issubset(endpoint["supported_parameters"]) and endpoint["status"] == 0]
    if not eligible:
        raise ValueError("no eligible endpoint supplies the pinned judge settings")
    rates = []
    for field in ("prompt", "completion"):
        values = []
        for endpoint in eligible:
            raw = endpoint["pricing"][field]
            if isinstance(raw, bool):
                raise ValueError("boolean token rate")
            rate = float(raw)
            if not valid_cost(rate):
                raise ValueError("token rate must be finite and nonnegative")
            values.append(rate * 1_000_000)
        rates.append(max(values))
    routing["max_price"] = dict(zip(("prompt", "completion"), rates, strict=True))
    global JUDGE_SOURCE, JUDGE_INPUT_USD_PER_MTOKEN, JUDGE_OUTPUT_USD_PER_MTOKEN
    JUDGE_SOURCE = f"{path.resolve()} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}"
    JUDGE_INPUT_USD_PER_MTOKEN, JUDGE_OUTPUT_USD_PER_MTOKEN = rates
    from . import adapters
    adapters.JUDGE_INPUT_USD_PER_MTOKEN, adapters.JUDGE_OUTPUT_USD_PER_MTOKEN = rates
    return {"model": JUDGE_MODEL, "input_usd_per_mtoken": rates[0], "output_usd_per_mtoken": rates[1], "source": JUDGE_SOURCE, "eligible_providers": sorted({endpoint["provider_name"] for endpoint in eligible}), "max_price": routing["max_price"]}


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
        4 * stage["item_count"] - stage.get("cached_request_count", 0)
        for stage in stages
        if stage["stage"] in {"target-aware-requests", "candidate-aware", "final-aware"}
    )
    blind = sum(
        2 * stage["item_count"] - stage.get("cached_request_count", 0)
        for stage in stages
        if stage["stage"] in {"blind-requests", "candidate-blind", "final-blind"}
    )
    return {
        "target_aware": target_aware,
        "blind": blind,
        "persona_validation": sum(stage["item_count"] - stage.get("cached_request_count", 0) for stage in stages if stage["stage"] == "persona-validation"),
    }


def _cost_lines(stages: list[dict]) -> tuple[list[BudgetLine], dict[str, int], int, int]:
    if JUDGE_SOURCE is None or JUDGE_INPUT_USD_PER_MTOKEN is None or JUDGE_OUTPUT_USD_PER_MTOKEN is None:
        raise RuntimeError("judge pricing is unsourced for deepseek/deepseek-v4-flash-0731; paid preflight is disabled")
    gpu_stages = sum(stage["runner"] == "modal_gpu" for stage in stages)
    gpu_hours = gpu_stages * GPU_HOURS_PER_STAGE
    request_counts = _request_counts(stages)
    input_tokens = (
        request_counts["target_aware"] * 4_000
        + request_counts["blind"] * 2_000
        + request_counts["persona_validation"] * 2_000
    )
    output_tokens = (
        (request_counts["target_aware"] + request_counts["blind"]) * 1_024
        + request_counts["persona_validation"] * 100
    )
    lines = [
        BudgetLine("Modal A10G GPU", gpu_hours, "GPU-hour", MODAL_A10G_USD_PER_GPU_HOUR, "USD/GPU-hour", MODAL_SOURCE),
        BudgetLine("Modal CPU", gpu_hours * 3600 * CPU_CORES_PER_GPU_STAGE, "core-second", MODAL_CPU_USD_PER_CORE_SECOND, "USD/core-second", MODAL_SOURCE),
        BudgetLine("Modal memory", gpu_hours * 3600 * MEMORY_GIB_PER_GPU_STAGE, "GiB-second", MODAL_MEMORY_USD_PER_GIB_SECOND, "USD/GiB-second", MODAL_SOURCE),
        BudgetLine("Judge input", input_tokens / 1_000_000, "M-token", JUDGE_INPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
        BudgetLine("Judge output", output_tokens / 1_000_000, "M-token", JUDGE_OUTPUT_USD_PER_MTOKEN, "USD/M-token", JUDGE_SOURCE),
    ]
    return lines, request_counts, input_tokens, output_tokens


def cost_estimate(stages: list[dict], *, retry_stages: list[dict] | None = None) -> dict:
    expected, request_counts, input_tokens, output_tokens = _cost_lines(stages)
    candidates = stages if retry_stages is None else retry_stages
    retry_stage = max(candidates, key=lambda stage: _sum(_cost_lines([stage])[0]), default=None)
    retry, retry_counts, retry_input_tokens, retry_output_tokens = _cost_lines([] if retry_stage is None else [retry_stage])
    gpu_stages = sum(stage["runner"] == "modal_gpu" for stage in stages)
    gpu_hours = gpu_stages * GPU_HOURS_PER_STAGE
    expected_usd = _sum(expected)
    retry_usd = _sum(retry)
    reservations = [
        {"kind": "expected_modal_and_judge_work", "upper_usd": expected_usd},
        {"kind": "one_affected_stage_retry", "upper_usd": retry_usd},
    ]
    return {
        "schema": "bsbench-budget-estimate-v3",
        "judge_model": JUDGE_MODEL,
        "planning_assumptions": {"gpu_hours_per_stage_upper": GPU_HOURS_PER_STAGE, "cpu_cores_per_gpu_stage": CPU_CORES_PER_GPU_STAGE, "memory_gib_per_gpu_stage": MEMORY_GIB_PER_GPU_STAGE, "modal_gpu_stage_upper_usd": MODAL_GPU_STAGE_UPPER_USD, "target_aware_input_tokens_per_request": 4_000, "blind_input_tokens_per_request": 2_000, "output_tokens_per_request": 1_024, "planned_attempts_per_request": 1, "attempts_per_request_upper": 3, "concurrency": 6},
        "quantities": {"gpu_stages": gpu_stages, "gpu_hours": gpu_hours, "requests": request_counts, "input_tokens": input_tokens, "output_tokens": output_tokens},
        "expected_work": [line.record() for line in expected],
        "retry_reserve": {
            "scope": "largest single corrected stage",
            "stage": retry_stage,
            "quantities": {"requests": retry_counts, "input_tokens": retry_input_tokens, "output_tokens": retry_output_tokens},
            "work": [line.record() for line in retry],
            "subtotal_usd": retry_usd,
        },
        "unresolved_reserve": [],
        "planned_reservations": reservations,
        "total_upper_usd": expected_usd + retry_usd,
    }


PHASE6_SMOKE_LEDGER = Path(__file__).parents[3] / "outputs" / "bsbench-smoke" / "costs.jsonl"


def phase6_smoke_committed() -> float:
    return committed(PHASE6_SMOKE_LEDGER)


def preflight_budget(ledger: Path, estimate: dict, *, external_committed_usd: float = 0.0) -> dict:
    if not valid_cost(external_committed_usd) or not valid_cost(estimate["total_upper_usd"]):
        raise ValueError("preflight requires finite nonnegative costs")
    existing_ledger_usd = committed(ledger)
    existing = existing_ledger_usd + external_committed_usd
    total = existing + estimate["total_upper_usd"]
    if total >= BUDGET_LIMIT_USD:
        raise RuntimeError(f"budget: ${total:.2f} is at or above ${BUDGET_LIMIT_USD:.2f}")
    return estimate | {
        "limit_usd": BUDGET_LIMIT_USD,
        "existing_committed_usd": existing,
        "existing_ledger_usd": existing_ledger_usd,
        "external_committed_usd": external_committed_usd,
        "total_upper_usd": total,
    }


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
    """Describe only the remaining 20-question evaluation and disjoint transfer work."""
    records = {PREDICTION_CASES[0].case_id: load_evaluation_records()} | load_transfer_records()
    item_count = sum(len(records[case.case_id]) for case in PREDICTION_CASES) * len(FINAL_DOSE_MULTIPLIERS) * 2
    provenance = transfer_records_identity(records)
    return tuple(
        {
            "stage": stage,
            "runner": runner,
            "method": method,
            "random_seed": random_seed,
            "item_count": len(records[EVALUATION_CASE.case_id]) * 6 if stage in {"final-aware", "final-blind"} else item_count,
            "prediction_provenance_sha256": content_key(provenance),
        }
        for method in METHODS
        if method not in {"bare", "prompting"}
        for random_seed in (RANDOM_SEEDS if method == "random" else (0,))
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
    transfer_cases: tuple[Case, ...] = PREDICTION_CASES,
) -> tuple[dict, ...]:
    """Describe the post-judgment generation graph without dispatching it."""
    if not vector_sha256 or not observed or not transfer_predictions or not case_prompts or not prompt_spec:
        raise ValueError("final stages require vector, observed records, transfer predictions, prompts and prompt spec")
    disjoint_cases = tuple(case for case in transfer_cases if case.case_id != EVALUATION_CASE.case_id)
    validate_cases(CALIBRATION_CASE, disjoint_cases)
    if any(case.case_id == EVALUATION_CASE.case_id and case != EVALUATION_CASE for case in transfer_cases):
        raise ValueError("evaluation case identity changed")
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
        if prediction.get("schema") != "bsbench-signed-rms-kl-transfer-v2":
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
        signed_predictions = prediction.get("signed_predictions")
        if not isinstance(signed_predictions, list) or len(signed_predictions) != 2 or {item.get("side") for item in signed_predictions} != {"+C", "-C"}:
            raise ValueError("final stages require one prediction for each side")
        if any(not math.isfinite(float(item.get("magnitude", 0))) or float(item.get("magnitude", 0)) <= 0 for item in signed_predictions):
            raise ValueError("final stages require finite positive signed magnitudes")
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
        "signed_candidate_doses": sorted(
            ({"magnitude": float(row["magnitude"]), "side": row["side"]} for row in ordered_observed),
            key=lambda row: (row["magnitude"], row["side"]),
        ),
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
            "item_count": sum(len(actual_case_prompts[plan["case"]["case_id"]]) * len(plan["coefficients"]) for plan in ordered_plans if plan["case"]["case_id"] == EVALUATION_CASE.case_id) if runner == "local_judge_api" else item_count,
            "input_stage": "final-generation",
        }
        for stage, runner in (
            ("final-health", "local"),
            ("final-aware", "local_judge_api"),
            ("final-blind", "local_judge_api"),
        )
    )
    return (generation, *downstream)


def persona_extraction_identity() -> dict:
    corpus = persona_corpus_identity(thinking=BSBENCH_PERSONA_THINKING)
    if corpus["actual_pairs"] != BSBENCH_PERSONA_ACTUAL_PAIRS or corpus["corpus_sha256"] != BSBENCH_PERSONA_CORPUS_SHA256:
        raise ValueError("the canonical persona corpus changed; update its reviewed identity deliberately")
    return {
        "pairs": [list(pair) for pair in BSBENCH_PERSONAS],
        "template": BSBENCH_PERSONA_TEMPLATE,
        "seed": BSBENCH_PERSONA_SEED,
        "requested_pairs": BSBENCH_PERSONA_REQUESTED_PAIRS,
        "actual_pairs": corpus["actual_pairs"],
        "corpus_sha256": corpus["corpus_sha256"],
        "thinking": BSBENCH_PERSONA_THINKING,
    }


def condition_stages(method: str) -> tuple[tuple[str, str], ...]:
    if method == "bare":
        return (("generation", "modal_gpu"), ("generation-health", "local"))
    if method == "prompting":
        return (("generation", "modal_gpu"), ("generation-health", "local"), ("target-aware-requests", "local_judge_api"), ("blind-requests", "local_judge_api"), ("persona-validation", "local_judge_api"))
    return (("calibration-candidates", "modal_gpu"), ("candidate-health", "local"), ("candidate-aware", "local_judge_api"))


def account_consumed_retry(estimate: dict, ledger: Path) -> dict:
    reservation = "7b44bbd07c9e5da2752e3933b462e4a592f68fd6747e261d3fe2cb580aa386d0"
    rows = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    original = [row for row in rows if row["event"] == "reserved" and row["id"] == reservation]
    if not original:
        return estimate
    reserved, = original
    resolution, = [row for row in rows if row["event"] == "estimated_at_reservation_upper" and row["reservation"] == reservation]
    if reserved["kind"] != "modal-calibration-candidates-random" or reserved["upper_usd"] != MODAL_GPU_STAGE_UPPER_USD or resolution["estimated_usd"] != reserved["upper_usd"]:
        raise ValueError("consumed retry allowance does not match the reconciled random failure")
    allowance = estimate["retry_reserve"]["subtotal_usd"]
    return estimate | {
        "total_upper_usd": estimate["total_upper_usd"] - allowance,
        "planned_reservations": [row | {"upper_usd": 0.0} if row["kind"] == "one_affected_stage_retry" else row for row in estimate["planned_reservations"]],
        "retry_reserve": {
            "scope": "original single-stage retry allowance, consumed",
            "additional_stage_estimate_not_reserved": estimate["retry_reserve"],
            "original_allowance_usd": reserved["upper_usd"],
            "consumed_reservation": reservation,
            "consumed_upper_usd": resolution["estimated_usd"],
            "subtotal_usd": 0.0,
            "remaining_reserve_usd": 0.0,
            "decision": "Parent 2026-09-22 seq13: original one-stage allowance consumed; no automatic replenishment; further stage failure requires review/rebudget.",
            "evidence": "slop/audits/20260922_random_config_attestation_failure.md",
        },
    }


def dry_manifest(out: Path, model_id: str = MODEL_ID, *, ledger: Path | None = None, cache_aware: bool = False, judge_endpoint: str = "https://openrouter.ai/api/v1/chat/completions") -> dict:
    rows = read_dev_cohort()
    prompts = [row["prompt"] for row in rows]
    prompts_by_id = {row["question_id"]: row["prompt"] for row in rows}
    calibration_prompts = [prompts_by_id[prompt_id] for prompt_id in CALIBRATION_CASE.prompt_ids]
    data = cohort_identity(rows)
    model = {"id": model_id}
    cache_root = out / "dry-plan-cache"
    ledger = out / "costs.jsonl" if ledger is None else ledger
    stages = []
    persona_source = persona_extraction_identity()
    for method, random_seed in ((method, seed) for method in METHODS for seed in (RANDOM_SEEDS if method == "random" else (0,))):
        vector_method = method not in {"bare", "prompting"}
        stage_prompts = calibration_prompts if vector_method else prompts
        config = {
            "condition": method,
            "random_seed": random_seed,
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
                    "random_seed": random_seed,
                    "item_count": PERSONA_VALIDATION_PAIRS if stage == "persona-validation" else 2 * len(calibration_prompts) * CANDIDATE_DOSE_UPPER if vector_method else len(stage_prompts),
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
    phase6_committed_usd = phase6_smoke_committed()
    manifest["external_commitments"] = [{
        "name": "phase6_modal_smoke",
        "ledger": str(PHASE6_SMOKE_LEDGER),
        "committed_usd": phase6_committed_usd,
    }]
    remaining_stages = stages + list(final_budget_stages)
    cache_hits = []
    if cache_aware:
        from .production import direct_cached_judge_counts, migrate_direct_generation, vector_cached_work
        direct_configs = {}
        for method in ("bare", "prompting"):
            direct_config = {"upper_usd": MODAL_GPU_STAGE_UPPER_USD, "prompt_spec": {"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128}, "condition": method, "prompt_ids": [row["question_id"] for row in rows]}
            if method == "prompting":
                direct_config |= {"persona_source": persona_source, "persona_validation_prompt_ids": list(PERSONA_VALIDATION_PROMPT_IDS)}
            direct_configs[method] = direct_config
            if migrate_direct_generation(out, model=model, data=data, method=method, config=direct_config, prompts=prompts):
                cache_hits.append({"stage": "generation", "method": method})
        remaining_stages = [stage for stage in remaining_stages if {"stage": stage["stage"], "method": stage["method"]} not in cache_hits]
        cached_judges = direct_cached_judge_counts(out, model=model | {"judge_model": JUDGE_MODEL}, data=data, rows=rows, configs=direct_configs, endpoint=judge_endpoint)
        manifest["validated_direct_judge_cache_counts"] = cached_judges
        for stage in remaining_stages:
            if stage["method"] == "prompting" and stage["stage"] in cached_judges:
                stage["cached_request_count"] = cached_judges[stage["stage"]]
        for method, seed in ((method, seed) for method in METHODS[2:] for seed in (RANDOM_SEEDS if method == "random" else (0,))):
            hits, magnitudes = vector_cached_work(out, model={"id": model_id, "judge_model": JUDGE_MODEL}, data=data, method=method, random_seed=seed, calibration_prompts=calibration_prompts, prompt_spec={"template": "Answer in 2 short sentences.", "enable_thinking": False, "max_new_tokens": 128}, judge_endpoint=judge_endpoint)
            cache_hits.extend({"stage": stage, "method": method, "random_seed": seed} for stage in sorted(hits))
            remaining_stages = [stage for stage in remaining_stages if not (stage["method"] == method and stage["random_seed"] == seed and stage["stage"] in hits)]
            for stage in remaining_stages:
                if stage["method"] == method and stage["random_seed"] == seed and stage["stage"].startswith("candidate-"):
                    stage["item_count"] = 2 * len(calibration_prompts) * magnitudes
    manifest["cache_aware"] = cache_aware
    manifest["validated_cache_hits"] = cache_hits
    manifest["remaining_stages"] = remaining_stages
    estimate = cost_estimate(remaining_stages)
    require_resolved_ledger(ledger)
    require_resolved_ledger(PHASE6_SMOKE_LEDGER)
    estimate = account_consumed_retry(estimate, ledger)
    existing = committed(ledger)
    estimate |= {"limit_usd": BUDGET_LIMIT_USD, "existing_ledger_usd": existing, "external_committed_usd": phase6_committed_usd, "existing_committed_usd": existing + phase6_committed_usd, "total_upper_usd": estimate["total_upper_usd"] + existing + phase6_committed_usd}
    estimate["paid_preflight_passed"] = estimate["total_upper_usd"] < BUDGET_LIMIT_USD
    manifest["cost_estimate"] = estimate
    save_json(out / "manifest.json", manifest)
    save_json(out / "cost-estimate.json", estimate)
    return manifest
