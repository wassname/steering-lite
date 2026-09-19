"""Write the non-paying BS-bench preflight or import a completed production stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_lite.benchmark.cache import committed
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.production import record_completed_stage, run_direct_condition, run_live_two_step, run_stages
from steering_lite.benchmark.sweep import CALIBRATION_CASE, MODEL_ID, dry_manifest


def import_recorded_smoke(out: Path, model_id: str) -> dict:
    """Import the already-completed Modal smoke; do not dispatch another GPU job."""
    summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
    if summary["model"] != model_id:
        raise ValueError(f"recorded smoke model {summary['model']!r} does not match {model_id!r}")
    rows = read_dev_cohort()[:1]
    if summary["question_id"] != rows[0]["question_id"]:
        raise ValueError("recorded smoke question does not match the fixed development cohort")
    model = {"id": model_id}
    data = cohort_identity(rows)
    prompts = [rows[0]["prompt"]]
    records = []
    for method in summary["conditions"]:
        records.append(record_completed_stage(
            out,
            stage="generation",
            model=model,
            data=data,
            method=method,
            config={"recorded_modal_app": summary["modal_app"], "question_id": summary["question_id"]},
            prompts=prompts,
            result=summary["generation_records"][method] | {"recorded_modal_smoke": summary, "method": method},
        ))
    return {"mode": "run-recorded-smoke", "records": records, "production_cache": str(out / "cache")}


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--import-smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("outputs/bsbench-v2"))
    parser.add_argument("--stage", choices=("bare", "prompting", "vjp_cache"), default="vjp_cache")
    parser.add_argument("--backend", choices=("recorded", "fake"))
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if args.dry_run:
        result = dry_manifest(args.out, args.model)
    elif args.import_smoke:
        result = import_recorded_smoke(args.out, args.model)
    else:
        if args.backend == "fake":
            rows = read_dev_cohort()
            calibration = [next(row["prompt"] for row in rows if row["question_id"] == prompt_id) for prompt_id in CALIBRATION_CASE.prompt_ids]
            class FakeBackend:
                """Deterministic offline backend; its records are explicitly not experiment results."""
                def __init__(self): self.calls = []
                def gpu(self, *, stage, method, config, prompts):
                    self.calls.append(stage)
                    if stage == "calibration-candidates":
                        coefficients = [0.2, 0.4]
                        return {"actual_usd": 0.0, "vector_bytes": b"offline-fake-vector-v1", "candidate_coefficients": coefficients, "candidate_items": [{"coefficient": coefficient, "prompt_index": index, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(), "response": f"Fake candidate {coefficient}/{index}."} for coefficient in coefficients for index, prompt in enumerate(prompts)], "method_config": {"fake": True}}
                    if stage == "final-generation":
                        artifact = config["vector_artifact"]
                        assert __import__("hashlib").sha256(Path(artifact["backend_path"]).read_bytes()).hexdigest() == artifact["sha256"]
                        return {"actual_usd": 0.0, "answers": ["Fake local final answer." for _ in config["executable_generation_plan"]], "plan_sha256": config["executable_plan_sha256"], "fake": True}
                    return {"actual_usd": 0.0, "answers": ["Fake local answer." for _ in prompts], "fake": True}
            backend = FakeBackend()
            ledger = args.ledger or args.out / "costs.jsonl"
            observed = [{"coefficient": coefficient, "useful": True, "coherent": True, "provenance": f"offline-fake-{coefficient}", "generation_health": {"rep": 0.0, "gen_len": 1, "steer_tail": "fake", "per_t_mean": [0.0], "per_t_p90": [0.0], "per_t_p95": [0.0], "per_t_max": [0.0], "per_t_n": [1]}} for coefficient in (0.2, 0.4)]
            vector_loader = lambda _artifact: __import__("types").SimpleNamespace(cfg=__import__("types").SimpleNamespace(coeff=0.0))
            def measure(vector, *_args, **_kwargs):
                return {"kl_rms": abs(vector.cfg.coeff) + 0.5, **observed[-1]["generation_health"]}
            def solver(_vector, _model, _tokenizer, prompts, **_kwargs):
                return 0.3 + 0.01 * len(prompts[0]), [{"coeff": 0.3, "kl_rms": 0.9}]
            common = dict(model={"id": args.model}, data=cohort_identity(rows), method=args.stage, backend=backend, prompt_spec={"template": "Answer in 2 short sentences.", "max_new_tokens": 8})
            inputs = common | (dict(calibration_prompts=calibration, candidate_judgments=observed, measure=measure, solver=solver, vector_loader=vector_loader) if args.stage == "vjp_cache" else dict(prompts=[row["prompt"] for row in rows]))
            runner = run_live_two_step if args.stage == "vjp_cache" else run_direct_condition
            first = runner(args.out, ledger, **inputs)
            calls_after_first = list(backend.calls)
            second = runner(args.out, ledger, **inputs)
            result = {"mode": "offline-fake-two-step", "not_experimental_results": True,
                      "first_backend_calls": calls_after_first, "immediate_rerun_backend_calls": backend.calls[len(calls_after_first):],
                      "first": first, "second": second}
        else:
            summary = json.loads(Path("slop/verification/20260919_phase6-modal-smoke-summary.json").read_text())
            if summary["model"] != args.model:
                raise ValueError("recorded backend model does not match the completed smoke")
            rows = read_dev_cohort()[:1]
            class RecordedBackend:
                def gpu(self, **_kwargs):
                    return summary["generation_records"][args.stage] | {"actual_usd": 0.0}
            ledger = args.ledger or args.out / "costs.jsonl"
            smoke_ledger = Path("outputs/bsbench-smoke/costs.jsonl")
            external = committed(smoke_ledger) if ledger != smoke_ledger else 0.0
            if committed(ledger) + external + 0.001 >= 50.0:
                raise RuntimeError("budget includes the unresolved smoke reservation")
            result = run_stages(args.out, ledger, [{"stage": "generation", "model": {"id": args.model}, "data": cohort_identity(rows), "method": args.stage, "config": {"upper_usd": 0.001, "recorded_modal_app": summary["modal_app"]}, "prompts": [rows[0]["prompt"]]}], RecordedBackend())
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
