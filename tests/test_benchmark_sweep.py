from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.cache import content_key
from steering_lite.benchmark.dose_search import (
    CALIBRATION_CASE,
    Case,
    TRANSFER_CASES,
    classify_transfer_boundary,
    fit_target,
    predict_transfer,
)
from steering_lite.benchmark.generation import read_dev_cohort
from steering_lite.benchmark.transfer_data import PromptRecord, load_transfer_records
from steering_lite.benchmark.pipeline import METHODS
from steering_lite.benchmark.sweep import (
    BSBENCH_PERSONAS,
    CANDIDATE_DOSE_UPPER,
    BSBENCH_PERSONA_ACTUAL_PAIRS,
    BSBENCH_PERSONA_CORPUS_SHA256,
    BSBENCH_PERSONA_REQUESTED_PAIRS,
    BSBENCH_PERSONA_SEED,
    BSBENCH_PERSONA_TEMPLATE,
    BSBENCH_PERSONA_THINKING,
    cached_dry_stage,
    cost_estimate,
    dry_manifest,
    final_stages,
    preflight_budget,
    persona_extraction_identity,
    reserve_budget,
)


def _method_stages(manifest: dict, method: str) -> list[dict]:
    return [stage for stage in manifest["stages"] if stage["method"] == method]


def _record(prompt_id: str, prompt: str, dataset: str = "synthetic") -> PromptRecord:
    answer_key = "The premise is false."
    return PromptRecord(prompt_id, prompt, dataset, "test/source", "test-revision", "test-source-hash", hashlib.sha256(prompt.encode()).hexdigest(), answer_key, hashlib.sha256(answer_key.encode()).hexdigest())


def _actual_transfer_cases() -> tuple[Case, ...]:
    return (
        Case("bsbench-v2-heldout-transfer", "synthetic", ("SYN-001",)),
        Case("other-dataset-transfer", "synthetic", ("SYN-002", "SYN-003")),
    )


def _prediction(case: Case, coefficient: float, *, target_id: str = "target-a", method: str = "vjp_cache") -> dict:
    return {
        "schema": "bsbench-rms-kl-transfer-v1",
        "target_id": target_id,
        "case": {"case_id": case.case_id, "dataset": case.dataset, "prompt_ids": list(case.prompt_ids)},
        "method": method,
        "model": "synthetic",
        "target_stat": "kl_rms",
        "target_rms": 1.25,
        "bracket": (0.1, 1.0),
        "predicted_coefficient": coefficient,
        "search_history": [],
    }


def _final_inputs() -> dict:
    transfer_cases = _actual_transfer_cases()
    return {
        "method": "vjp_cache",
        "vector_sha256": "vector-a",
        "observed": [
            {"coefficient": 0.4, "useful": True, "coherent": True, "provenance": "judge-b"},
            {"coefficient": -0.2, "useful": True, "coherent": True, "provenance": "judge-a"},
        ],
        "transfer_predictions": [
            _prediction(transfer_cases[0], 0.5),
            _prediction(transfer_cases[1], -0.25),
        ],
        "case_prompts": {
            transfer_cases[0].case_id: [_record("SYN-001", "heldout prompt")],
            transfer_cases[1].case_id: [_record("SYN-002", "other prompt one"), _record("SYN-003", "other prompt two")],
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

    assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in _method_stages(first, "bare")] == [
        ("generation", "modal_gpu", 20),
        ("generation-health", "local", 20),
    ]
    assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in _method_stages(first, "prompting")] == [
        ("generation", "modal_gpu", 20),
        ("generation-health", "local", 20),
        ("target-aware-requests", "local_judge_api", 20),
        ("blind-requests", "local_judge_api", 20),
    ]
    assert all(stage["config"]["persona_source"] is None for method in ("bare", "prompting") for stage in _method_stages(first, method))

    vector_methods = set(METHODS) - {"bare", "prompting"}
    assert len(vector_methods) == 6
    expected_persona_source = {
        "pairs": [list(pair) for pair in BSBENCH_PERSONAS],
        "template": BSBENCH_PERSONA_TEMPLATE,
        "seed": BSBENCH_PERSONA_SEED,
        "requested_pairs": BSBENCH_PERSONA_REQUESTED_PAIRS,
        "actual_pairs": BSBENCH_PERSONA_ACTUAL_PAIRS,
        "corpus_sha256": BSBENCH_PERSONA_CORPUS_SHA256,
        "thinking": BSBENCH_PERSONA_THINKING,
    }
    for method in vector_methods:
        stages = _method_stages(first, method)
        assert [(stage["stage"], stage["runner"], stage["item_count"]) for stage in stages] == [
            ("calibration-candidates", "modal_gpu", 4 * CANDIDATE_DOSE_UPPER),
            ("candidate-health", "local", 4 * CANDIDATE_DOSE_UPPER),
            ("candidate-aware", "local_judge_api", 4 * CANDIDATE_DOSE_UPPER),
            ("candidate-blind", "local_judge_api", 4 * CANDIDATE_DOSE_UPPER),
        ]
        assert {stage["prompts_sha256"] for stage in stages} == {
            content_key({"prompts": calibration_prompts})
        }
        assert all(stage["config"]["calibration_case"]["prompt_ids"] == list(CALIBRATION_CASE.prompt_ids) for stage in stages)
        assert all(stage["config"]["calibration_prompts_sha256"] == content_key({"prompts": calibration_prompts}) for stage in stages)
        assert all(stage["config"]["persona_source"] == expected_persona_source for stage in stages)
        assert all(stage["config"]["persona_source_sha256"] == content_key(expected_persona_source) for stage in stages)
        assert all(stage["config"]["candidate_dose_upper"] == CANDIDATE_DOSE_UPPER for stage in stages)

    phase_b = first["phase_b_budget_stages"]
    assert len(phase_b) == 24
    assert {(stage["stage"], stage["runner"], stage["item_count"]) for stage in phase_b} == {
        ("final-generation", "modal_gpu", 24),
        ("final-health", "local", 24),
        ("final-aware", "local_judge_api", 24),
        ("final-blind", "local_judge_api", 24),
    }
    assert all("dispatch_blocked_by_missing_transfer_data" not in stage for stage in phase_b)
    assert all("transfer_provenance_sha256" in stage for stage in phase_b)
    assert estimate["quantities"]["gpu_stages"] == 14
    assert estimate["quantities"]["requests"] == {
        "target_aware": 904,
        "blind": 904,
        "persona_validation": 12,
    }
    assert estimate["judge_model"] == first["judge_model"] == "deepseek/deepseek-chat"
    assert estimate["total_upper_usd"] < 50.0
    assert first["paid_execution_enabled"] is False
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "cost-estimate.json").exists()
    assert (tmp_path / "dry-plan-cache").exists()
    assert not (tmp_path / "cache").exists()


def test_persona_identity_distinguishes_requested_and_actual_and_rejects_corpus_drift(monkeypatch):
    identity = persona_extraction_identity()
    assert identity["requested_pairs"] == 256
    assert identity["actual_pairs"] == 200
    assert identity["corpus_sha256"] == BSBENCH_PERSONA_CORPUS_SHA256
    monkeypatch.setattr("steering_lite.benchmark.sweep.persona_corpus_identity", lambda **_kwargs: {"actual_pairs": 199, "corpus_sha256": "changed", "thinking": True})
    with pytest.raises(ValueError, match="canonical persona corpus changed"):
        persona_extraction_identity()


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
        ("final-generation", "modal_gpu", 9),
        ("final-health", "local", 9),
        ("final-aware", "local_judge_api", 9),
        ("final-blind", "local_judge_api", 9),
    ]
    assert stages == reordered
    assert stages[0]["case_prompts"] == {case_id: [record.prompt for record in records] for case_id, records in inputs["case_prompts"].items()}
    assert stages[0]["case_prompt_records"]["bsbench-v2-heldout-transfer"][0]["source_revision"] == "test-revision"
    assert stages[0]["generation_plan"] == [
        {"case": {"case_id": "bsbench-v2-heldout-transfer", "dataset": "synthetic", "prompt_ids": ["SYN-001"]}, "prompts": ["heldout prompt"], "coefficients": [0.4, 0.5, 0.6]},
        {"case": {"case_id": "other-dataset-transfer", "dataset": "synthetic", "prompt_ids": ["SYN-002", "SYN-003"]}, "prompts": ["other prompt one", "other prompt two"], "coefficients": [-0.2, -0.25, -0.3]},
    ]
    assert all("case_prompts" not in stage and "generation_plan" not in stage for stage in stages[1:])
    assert all(stage["input_stage"] == "final-generation" for stage in stages[1:])
    config = stages[0]["config"]
    assert config["vector_sha256"] == "vector-a"
    assert config["candidate_coefficients"] == [-0.2, 0.4]
    assert config["prompt_spec"] == inputs["prompt_spec"]
    assert config["prompt_spec_sha256"] == content_key(inputs["prompt_spec"])
    assert config["case_prompt_hashes"] == {
        case_id: content_key({"prompts": [record.prompt for record in records]})
        for case_id, records in inputs["case_prompts"].items()
    }
    assert all(stage["config"] == config for stage in stages)


def test_final_stage_identity_invalidates_each_input_independently():
    inputs = _final_inputs()
    baseline = final_stages(**inputs)[0]["config"]
    changes = (
        {"vector_sha256": "vector-b"},
        {"observed": [{**inputs["observed"][0], "provenance": "different"}, inputs["observed"][1]]},
        {"observed": [{**inputs["observed"][0], "coefficient": 0.5}, inputs["observed"][1]]},
        {"transfer_predictions": [{**inputs["transfer_predictions"][0], "predicted_coefficient": 0.6}, inputs["transfer_predictions"][1]]},
        {"case_prompts": inputs["case_prompts"] | {"other-dataset-transfer": [replace(inputs["case_prompts"]["other-dataset-transfer"][0], source_revision="changed"), inputs["case_prompts"]["other-dataset-transfer"][1]]}},
        {"prompt_spec": inputs["prompt_spec"] | {"template": "Answer directly."}},
    )
    for changed in changes:
        assert final_stages(**(inputs | changed))[0]["config"] != baseline


def test_final_stages_reject_missing_or_placeholder_inputs_before_dispatch():
    inputs = _final_inputs()
    for changed, message in (
        ({"vector_sha256": ""}, "vector"),
        ({"observed": []}, "observed"),
        ({"transfer_predictions": []}, "transfer predictions"),
        ({"case_prompts": {}}, "prompts"),
        ({"prompt_spec": {}}, "prompt spec"),
        ({"case_prompts": {"bsbench-v2-heldout-transfer": [_record("SYN-001", "only one")] }}, "complete prompt records"),
    ):
        with pytest.raises(ValueError, match=message):
            final_stages(**(inputs | changed))
    with pytest.raises(ValueError, match="auditable PromptRecord"):
        final_stages(**(inputs | {"case_prompts": {case.case_id: ["placeholder"] for case in inputs["transfer_cases"]}}))

    records = load_transfer_records()
    default_stages = final_stages(
        **(inputs | {
            "transfer_cases": TRANSFER_CASES,
            "transfer_predictions": [_prediction(case, 0.5) for case in TRANSFER_CASES],
            "case_prompts": records,
        })
    )
    assert default_stages[0]["item_count"] == 24

    predictions = inputs["transfer_predictions"]
    for changed, message in (
        ({"transfer_predictions": [{**predictions[0], "schema": "wrong"}, predictions[1]]}, "RMS-KL"),
        ({"transfer_predictions": [{**predictions[0], "method": "pca"}, predictions[1]]}, "requested method"),
        ({"transfer_predictions": [{**predictions[0], "case": {"case_id": predictions[0]["case"]["case_id"]}}, predictions[1]]}, "case identities"),
        ({"transfer_predictions": [predictions[0], {**predictions[1], "target_id": "other-target"}]}, "common transfer target"),
        ({"transfer_predictions": [{**predictions[0], "predicted_coefficient": float("nan")}, predictions[1]]}, "finite non-zero"),
        ({"transfer_predictions": [predictions[0], predictions[0]]}, "one transfer prediction"),
    ):
        with pytest.raises(ValueError, match=message):
            final_stages(**(inputs | changed))


def test_synthetic_calibration_transfer_flow_predicts_before_post_generation_classification():
    observed = [
        {"coefficient": 0.2, "useful": True, "coherent": True, "provenance": "candidate-1", "generation_health": {}},
        {"coefficient": 0.4, "useful": True, "coherent": True, "provenance": "candidate-2", "generation_health": {}},
        {"coefficient": 0.8, "useful": False, "coherent": False, "provenance": "candidate-3", "generation_health": {}},
    ]
    health = {
        "kl_rms": 1.25, "rep": 0.0, "gen_len": 20, "steer_tail": "calibration",
        "per_t_mean": [0.1], "per_t_p90": [0.1], "per_t_p95": [0.1],
        "per_t_max": [0.1], "per_t_n": [1],
    }
    calibration_prompts = ["calibration-only prompt"]
    transfer_prompts = ["disjoint transfer prompt one", "disjoint transfer prompt two"]
    measured_prompts = []
    solved_prompts = []

    def measure(*args, **_kwargs):
        measured_prompts.append(args[3])
        return health

    def solver(*args, **_kwargs):
        solved_prompts.append(args[3])
        return 0.7, [{"coeff": 0.7, "kl_rms": 1.25, "final": True}]

    vector = SimpleNamespace(cfg=SimpleNamespace(coeff=0.0))
    target = fit_target(
        vector, object(), object(), calibration_prompts, CALIBRATION_CASE, observed,
        method="vjp_cache", model_id="synthetic", measure_kwargs={}, measure=measure,
    )
    transfer_case = Case("synthetic-transfer", "audited-synthetic", ("SYN-001", "SYN-002"))
    prediction = predict_transfer(
        vector, object(), object(), transfer_prompts, target, transfer_case,
        bracket=(0.1, 1.0), solver_kwargs={}, solver=solver,
    )
    stages = final_stages(
        method="vjp_cache", vector_sha256="synthetic-vector", observed=observed,
        transfer_predictions=[prediction], case_prompts={transfer_case.case_id: [_record(f"SYN-{number:03d}", prompt, "audited-synthetic") for number, prompt in enumerate(transfer_prompts, 1)]},
        prompt_spec={"max_new_tokens": 8}, transfer_cases=(transfer_case,),
    )
    post_generation = classify_transfer_boundary(
        prediction,
        [
            {"case_id": transfer_case.case_id, "target_id": target["target_id"], "coefficient": coefficient, "useful": True, "coherent": True, "provenance": "transfer-1", "generation_health": {}}
            for coefficient in (0.56, 0.7, 0.84)
        ],
    )
    changed_prediction = prediction | {"predicted_coefficient": 0.6}
    changed_stages = final_stages(
        method="vjp_cache", vector_sha256="synthetic-vector", observed=observed,
        transfer_predictions=[changed_prediction], case_prompts={transfer_case.case_id: [_record(f"SYN-{number:03d}", prompt, "audited-synthetic") for number, prompt in enumerate(transfer_prompts, 1)]},
        prompt_spec={"max_new_tokens": 8}, transfer_cases=(transfer_case,),
    )
    retargeted_stages = final_stages(
        method="vjp_cache", vector_sha256="synthetic-vector", observed=observed,
        transfer_predictions=[prediction | {"target_id": "changed-target"}],
        case_prompts={transfer_case.case_id: [_record(f"SYN-{number:03d}", prompt, "audited-synthetic") for number, prompt in enumerate(transfer_prompts, 1)]}, prompt_spec={"max_new_tokens": 8},
        transfer_cases=(transfer_case,),
    )

    assert measured_prompts == [calibration_prompts]
    assert solved_prompts == [transfer_prompts]
    assert target["observed_boundary"]["coefficient"] == 0.4
    assert prediction["predicted_coefficient"] == 0.7
    assert prediction["predicted_coefficient"] != target["source"]["coefficient"]
    assert prediction["case"]["prompt_ids"] != list(CALIBRATION_CASE.prompt_ids)
    assert "boundary" not in prediction
    assert stages[0]["item_count"] == 6
    assert stages[0]["generation_plan"] == [{
        "case": {"case_id": transfer_case.case_id, "dataset": transfer_case.dataset, "prompt_ids": list(transfer_case.prompt_ids)},
        "prompts": transfer_prompts,
        "coefficients": [0.56, 0.7, 0.84],
    }]
    assert all(stage["item_count"] == 6 for stage in stages)
    assert post_generation["boundary"] == "measured_useful_coherent_boundary"
    assert changed_stages[0]["config"] != stages[0]["config"]
    assert retargeted_stages[0]["config"] != stages[0]["config"]


def test_costs_use_per_stage_counts_and_blind_filtering(tmp_path: Path):
    manifest = dry_manifest(tmp_path)
    full = cost_estimate(manifest["stages"])
    no_gpu = cost_estimate([stage for stage in manifest["stages"] if stage["runner"] != "modal_gpu"])
    no_blind = cost_estimate([stage for stage in manifest["stages"] if "blind" not in stage["stage"]])

    assert no_gpu["total_upper_usd"] < full["total_upper_usd"]
    assert no_blind["quantities"]["requests"] == {
        "target_aware": 616,
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


def test_transfer_records_are_exact_provenanced_and_disjoint_from_calibration_and_eval():
    from steering_lite.benchmark.transfer_data import (
        BSBENCH_SOURCE_REVISION,
        BSBENCH_SOURCE_SHA256,
        PAPER_NATIVE_SOURCE_REVISION,
        PAPER_NATIVE_SOURCE_SHA256,
        TRANSFER_CASE_PROMPT_IDS,
    )

    records = load_transfer_records()
    assert set(records) == set(TRANSFER_CASE_PROMPT_IDS)
    assert {case.case_id: tuple(record.prompt_id for record in records[case.case_id]) for case in TRANSFER_CASES} == TRANSFER_CASE_PROMPT_IDS
    protected = read_dev_cohort()
    protected_ids = {row["question_id"] for row in protected}
    protected_text = {row["prompt"] for row in protected}
    assert not protected_ids.intersection(record.prompt_id for case_records in records.values() for record in case_records)
    assert not protected_text.intersection(record.prompt for case_records in records.values() for record in case_records)
    assert all(record.content_sha256 == hashlib.sha256(record.prompt.encode()).hexdigest() for case_records in records.values() for record in case_records)
    bsbench = records["bsbench-v2-heldout-a"]
    native = records["paper-native-false-claim-agreement-a"]
    assert {(record.source_path, record.source_revision, record.source_sha256) for record in bsbench} == {
        ("data/bullshit_bench_v2.jsonl", BSBENCH_SOURCE_REVISION, BSBENCH_SOURCE_SHA256)
    }
    assert {(record.source_path, record.source_revision, record.source_sha256) for record in native} == {
        ("data/dev/paper_native_false_claim_agreement.json", PAPER_NATIVE_SOURCE_REVISION, PAPER_NATIVE_SOURCE_SHA256)
    }
