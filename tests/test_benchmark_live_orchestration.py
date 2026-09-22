import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from steering_lite.benchmark.dose_search import CALIBRATION_CASE
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.production import persona_source_identity, run_direct_condition, run_live_two_step
from steering_lite.benchmark.sweep import MODAL_GPU_STAGE_UPPER_USD, dry_manifest
from steering_lite.benchmark.transfer_data import load_transfer_records


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.configs = []

    def gpu(self, *, stage, method, config, prompts):
        self.calls.append(stage)
        self.configs.append(config)
        if stage == "calibration-candidates":
            magnitudes = [0.2, 0.4]
            return {"actual_usd": 0.0, "vector_bytes": b"durable-test-vector", "candidate_magnitudes": magnitudes, "candidate_items": [{"magnitude": magnitude, "side": side, "prompt_index": index, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(), "response": f"answer-{side}-{magnitude}-{index}"} for magnitude in magnitudes for side in ("+C", "-C") for index, prompt in enumerate(prompts)], "method_config": {"method": method, "layers": [7, 11, 15, 19, 23], "seed": 0, "target_layer": 29, "skip_first": 16}}
        if stage == "final-generation":
            artifact = config["vector_artifact"]
            assert "backend_path" not in artifact
            assert __import__("base64").b64decode(artifact["vector_bytes_b64"]) == b"durable-test-vector"
            return {"actual_usd": 0.0, "answers": ["answer." for _ in config["executable_generation_plan"]], "plan_sha256": config["executable_plan_sha256"]}
        return {"actual_usd": 0.0, "answers": ["answer." for _ in prompts]}


def inputs(backend):
    rows = read_dev_cohort()
    prompts = [next(row["prompt"] for row in rows if row["question_id"] == prompt_id) for prompt_id in CALIBRATION_CASE.prompt_ids]
    health = {"rep": 0.0, "gen_len": 1, "steer_tail": "fake", "per_t_mean": [0.0], "per_t_p90": [0.0], "per_t_p95": [0.0], "per_t_max": [0.0], "per_t_n": [1]}
    observed = [{"magnitude": magnitude, "coefficient": magnitude, "side": side, "provenance": f"fake-{magnitude}-{side}", "generation_health": {"reasons": []}} for magnitude in (0.2, 0.4) for side in ("+C", "-C")]
    measured, solved = [], []
    def measure(vector, *_args, **_kwargs):
        measured.append(vector.cfg.coeff)
        return {"kl_rms": abs(vector.cfg.coeff) + 0.5, "n_pos": 4, **health}
    def solver(_vector, _model, _tokenizer, case_prompts, **kwargs):
        solved.append(tuple(case_prompts))
        magnitude = round(0.3 + ((len(solved) + 1) // 2) / 100, 2)
        coefficient = kwargs["sign"] * magnitude
        return coefficient, [{"coeff": coefficient, "kl_rms": 0.9}]
    return dict(model={"id": "fake"}, data=cohort_identity(rows), method="vjp_cache", calibration_prompts=prompts, backend=backend, prompt_spec={"max_new_tokens": 8}, candidate_judgments=observed, measure=measure, solver=solver, vector_loader=lambda _artifact: SimpleNamespace(cfg=SimpleNamespace(coeff=0.0))), measured, solved


def test_live_two_step_uses_real_calibration_functions_plan_and_full_cache(tmp_path: Path):
    backend = FakeBackend(); kwargs, measured, solved = inputs(backend)
    first = run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    assert backend.calls == ["calibration-candidates", "final-generation"]
    assert measured == [0.4, -0.4] and len(solved) == 10
    assert [item["signed_predictions"][0]["magnitude"] for item in first["transfer_prediction"]["predictions"]] == [0.31, 0.32, 0.33, 0.34, 0.35]
    assert len(first["final"]["answers"]) == len(first["final_aware"]["records"]) == 168
    assert all(row["fake"] and row["non_experimental"] and row["response"] for row in first["final_blind"]["records"])
    canonical = {plan["case"]["case_id"]: plan["coefficients"] for plan in first["final_stages"][0]["config"]["final_dose_plans"]}
    assert {case_id: [{"side": row["side"], "magnitude": row["magnitude"]} for row in first["final_aware"]["records"] if row["case_id"] == case_id][:6] for case_id in canonical} == canonical
    artifact = tmp_path / first["candidate"]["vector_artifact"]["path"]
    assert artifact.is_file() and first["candidate"]["vector_sha256"] == first["candidate"]["vector_artifact"]["sha256"]
    run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    assert backend.calls == ["calibration-candidates", "final-generation"]
    final_cache, = (tmp_path / "cache" / "final-generation").glob("*.json")
    persisted_config = json.loads(final_cache.read_text())["identity"]["config"]
    assert "backend_path" not in json.dumps(persisted_config)
    final_dispatch_config = backend.configs[-1]
    assert "backend_path" not in final_dispatch_config["vector_artifact"]
    assert __import__("base64").b64decode(final_dispatch_config["vector_artifact"]["vector_bytes_b64"]) == artifact.read_bytes()


def test_wrong_signed_method_config_fails_before_settlement(tmp_path: Path):
    class WrongConfig(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "calibration-candidates":
                result["method_config"] = result["method_config"] | {"layers": [3]}
            return result

    backend = WrongConfig(); kwargs, _, _ = inputs(backend); ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="method_config"):
        run_live_two_step(tmp_path, ledger, **kwargs)
    assert backend.calls == ["calibration-candidates"]
    assert not any('"event": "settled"' in line for line in ledger.read_text().splitlines())


def test_old_compatible_one_sided_candidate_cache_is_not_reused(tmp_path: Path):
    backend = FakeBackend(); kwargs, _, _ = inputs(backend); ledger = tmp_path / "ledger.jsonl"
    run_live_two_step(tmp_path, ledger, **kwargs)
    cache_path, = (tmp_path / "cache" / "calibration-candidates").glob("*.json")
    record = json.loads(cache_path.read_text())
    old_identity = record["identity"] | {"code_sha256": "d2a8eb38eebf8b090ae0eb66c73bb9b766b3e3762860e440089529385633ee88"}
    old_result = record["result"] | {"candidate_items": [item for item in record["result"]["candidate_items"] if item["side"] == "+C"]}
    cache_path.unlink()
    old_path = cache_path.parent / f"{__import__('hashlib').sha256(json.dumps(old_identity, sort_keys=True, allow_nan=False).encode()).hexdigest()}.json"
    old_path.write_text(json.dumps({"identity": old_identity, "result": old_result}))
    run_live_two_step(tmp_path, ledger, **kwargs)
    assert backend.calls.count("calibration-candidates") == 2


def test_candidate_item_coverage_and_observation_coefficients_fail_before_target(tmp_path: Path):
    class BrokenCandidate(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "calibration-candidates":
                result["candidate_items"] = result["candidate_items"][:-1]
            return result
    backend = BrokenCandidate(); kwargs, _, _ = inputs(backend)
    with pytest.raises(ValueError, match="cover every"):
        run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    assert backend.calls == ["calibration-candidates"]


@pytest.mark.parametrize("mutation", (lambda rows: rows[:-1], lambda rows: rows + [rows[0]]))
def test_candidate_item_missing_or_extra_fails_before_settlement(tmp_path: Path, mutation):
    class BrokenCandidate(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "calibration-candidates": result["candidate_items"] = mutation(result["candidate_items"])
            return result
    backend = BrokenCandidate(); kwargs, _, _ = inputs(backend)
    with pytest.raises(ValueError, match="cover every"):
        run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    assert backend.calls == ["calibration-candidates"]
    assert not any('"event": "settled"' in line for line in (tmp_path / "ledger.jsonl").read_text().splitlines())


def test_missing_final_or_direct_answers_fail_before_settlement(tmp_path: Path):
    class ShortAnswers(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] in {"final-generation", "generation"}: result["answers"] = result["answers"][:-1]
            return result
    backend = ShortAnswers(); kwargs, _, _ = inputs(backend)
    with pytest.raises(ValueError, match="one answer"):
        run_live_two_step(tmp_path / "vector", tmp_path / "vector-ledger.jsonl", **kwargs)
    with pytest.raises(ValueError, match="one answer"):
        run_direct_condition(tmp_path / "direct", tmp_path / "direct-ledger.jsonl", model={"id": "fake"}, data={"sha256": "fake"}, method="bare", prompts=["one", "two"], backend=backend, prompt_spec={"max_new_tokens": 8})


def test_over_limit_candidate_doses_fail_before_settlement(tmp_path: Path):
    from steering_lite.benchmark.sweep import CANDIDATE_DOSE_UPPER
    class TooManyDoses(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            if kwargs["stage"] == "calibration-candidates":
                magnitudes = [round(0.01 * number, 2) for number in range(1, CANDIDATE_DOSE_UPPER + 2)]
                result["candidate_magnitudes"] = magnitudes
                result["candidate_items"] = [{"magnitude": magnitude, "side": side, "prompt_index": index, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(), "response": "answer"} for magnitude in magnitudes for side in ("+C", "-C") for index, prompt in enumerate(kwargs["prompts"])]
            return result
    backend = TooManyDoses(); kwargs, _, _ = inputs(backend); ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(ValueError, match="unique positive candidate magnitudes"):
        run_live_two_step(tmp_path, ledger, **kwargs)
    assert backend.calls == ["calibration-candidates"]
    assert not any('"event": "settled"' in line for line in ledger.read_text().splitlines())


def test_missing_or_corrupt_vector_fails_before_final_dispatch(tmp_path: Path):
    backend = FakeBackend(); kwargs, _, _ = inputs(backend)
    with pytest.raises(ValueError, match="explicit candidate"):
        run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **(kwargs | {"candidate_judgments": None}))
    with pytest.raises(ValueError, match="magnitude/side cells"):
        run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **(kwargs | {"candidate_judgments": [kwargs["candidate_judgments"][0]]}))
    result = run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    (tmp_path / result["candidate"]["vector_artifact"]["path"]).unlink()
    before = list(backend.calls)
    with pytest.raises(RuntimeError, match="sidecar"):
        run_live_two_step(tmp_path, tmp_path / "ledger.jsonl", **kwargs)
    assert backend.calls == before


def test_invalidation_boundaries(tmp_path: Path):
    backend = FakeBackend(); kwargs, _, _ = inputs(backend); ledger = tmp_path / "ledger.jsonl"
    run_live_two_step(tmp_path, ledger, **kwargs)
    changed_judgments = [{**row, "provenance": "changed"} if index == 1 else row for index, row in enumerate(kwargs["candidate_judgments"])]
    run_live_two_step(tmp_path, ledger, **(kwargs | {"candidate_judgments": changed_judgments}))
    assert backend.calls == ["calibration-candidates", "final-generation", "final-generation"]
    records = load_transfer_records()
    case = next(iter(records)); changed_records = records | {case: (replace(records[case][0], source_revision="changed"), *records[case][1:])}
    run_live_two_step(tmp_path, ledger, **(kwargs | {"transfer_records": changed_records}))
    assert backend.calls == ["calibration-candidates", "final-generation", "final-generation", "final-generation"]
    extraction = persona_source_identity() | {"test_extraction_revision": "two"}
    run_live_two_step(tmp_path, ledger, **(kwargs | {"extraction_identity": extraction}))
    assert backend.calls[-2:] == ["calibration-candidates", "final-generation"]


def test_fake_production_reservations_use_shared_dry_stage_upper(tmp_path: Path, monkeypatch):
    import steering_lite.benchmark.sweep as sweep_module

    monkeypatch.setattr(sweep_module, "JUDGE_SOURCE", "synthetic-test-pricing")
    monkeypatch.setattr(sweep_module, "JUDGE_INPUT_USD_PER_MTOKEN", 0.0001)
    monkeypatch.setattr(sweep_module, "JUDGE_OUTPUT_USD_PER_MTOKEN", 0.0001)
    backend = FakeBackend()
    rows = read_dev_cohort()
    direct = run_direct_condition(
        tmp_path / "direct",
        tmp_path / "direct-ledger.jsonl",
        model={"id": "fake"},
        data=cohort_identity(rows[:2]),
        method="bare",
        prompts=[row["prompt"] for row in rows[:2]],
        backend=backend,
        prompt_spec={"max_new_tokens": 8},
    )
    kwargs, _, _ = inputs(backend)
    live = run_live_two_step(tmp_path / "vector", tmp_path / "vector-ledger.jsonl", **kwargs)

    assert direct["generation"]["reused"] is False
    assert live["candidate"]["reused"] is False and live["final"]["reused"] is False
    assert [config["upper_usd"] for config in backend.configs] == [
        MODAL_GPU_STAGE_UPPER_USD,
        MODAL_GPU_STAGE_UPPER_USD,
        MODAL_GPU_STAGE_UPPER_USD,
    ]
    manifest = dry_manifest(tmp_path / "dry")
    assert manifest["cost_estimate"]["planning_assumptions"]["modal_gpu_stage_upper_usd"] == MODAL_GPU_STAGE_UPPER_USD


def test_overage_is_auditable_not_cached_and_blocks_retry(tmp_path: Path):
    class OverageBackend(FakeBackend):
        def gpu(self, **kwargs):
            result = super().gpu(**kwargs)
            return result | {"actual_usd": MODAL_GPU_STAGE_UPPER_USD + 0.01}

    backend = OverageBackend()
    kwargs = {
        "model": {"id": "fake"},
        "data": {"sha256": "fake"},
        "method": "bare",
        "prompts": ["one"],
        "backend": backend,
        "prompt_spec": {"max_new_tokens": 8},
    }
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(RuntimeError, match="exceeded reservation"):
        run_direct_condition(tmp_path, ledger, **kwargs)
    assert backend.calls == ["generation"]
    assert not list((tmp_path / "cache" / "generation").glob("*.json"))
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [record["event"] for record in records] == ["reserved", "settled", "overage"]
    assert records[-1]["actual_usd"] > records[-1]["upper_usd"]

    with pytest.raises(RuntimeError, match="unresolved overage"):
        run_direct_condition(tmp_path, ledger, **kwargs)
    assert backend.calls == ["generation"]


class DirectJudge:
    endpoint = "offline-direct-judge"

    def __init__(self):
        self.requests = []

    def complete(self, requests):
        self.requests.extend(requests)
        return [
            {"summary": "difference", "changes": []}
            if request["blind"]
            else {"on_axis_A": 0.0, "on_axis_B": 1.0, "off_axis_A": 0.0, "off_axis_B": 0.0}
            for request in requests
        ]


def test_bare_is_cached_origin_without_self_judgment(tmp_path: Path):
    backend = FakeBackend(); rows = read_dev_cohort()[:2]
    result = run_direct_condition(tmp_path, tmp_path / "ledger.jsonl", model={"id": "fake"}, data=cohort_identity(rows), method="bare", prompts=[row["prompt"] for row in rows], rows=rows, backend=backend, prompt_spec={"max_new_tokens": 8})
    assert backend.calls == ["generation"]
    assert set(result) == {"paid_execution_enabled", "generation", "health", "baseline_answers"}
    assert [record["question_id"] for record in result["health"]["records"]] == [row["question_id"] for row in rows]
    run_direct_condition(tmp_path, tmp_path / "ledger.jsonl", model={"id": "fake"}, data=cohort_identity(rows), method="bare", prompts=[row["prompt"] for row in rows], rows=rows, backend=backend, prompt_spec={"max_new_tokens": 8})
    assert backend.calls == ["generation"]


def test_prompting_reuses_bare_and_persists_paired_judgments(tmp_path: Path):
    backend = FakeBackend(); rows = read_dev_cohort()[:2]; judge = DirectJudge()
    common = dict(model={"id": "fake", "judge_model": "fake-judge"}, data=cohort_identity(rows), method="prompting", prompts=[row["prompt"] for row in rows], rows=rows, backend=backend, prompt_spec={"max_new_tokens": 8}, judge=judge)
    bare = run_direct_condition(tmp_path, tmp_path / "ledger.jsonl", **(common | {"method": "bare", "judge": None}))
    result = run_direct_condition(tmp_path, tmp_path / "ledger.jsonl", **common)
    assert backend.calls == ["generation", "generation"]
    assert result["baseline_answers"] == bare["generation"]["answers"]
    assert len(result["judgments"]["requests"]) == len(result["judgments"]["responses"]) == 12
    assert len(result["aware"]["records"]) == 8
    assert len(result["blind"]["records"]) == 4
    assert all(not record["blind"] for record in result["aware"]["records"])
    assert all(record["blind"] for record in result["blind"]["records"])
    assert all("prompting" not in str(request["payload"]) and "+C" not in str(request["payload"]) for request in result["judgments"]["requests"] if request["blind"])
    run_direct_condition(tmp_path, tmp_path / "ledger.jsonl", **common)
    assert backend.calls == ["generation", "generation"] and len(judge.requests) == 12
