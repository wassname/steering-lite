"""Write the non-paying BS-bench preflight or import a completed production stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_lite.benchmark.cache import committed
from steering_lite.benchmark.generation import cohort_identity, read_dev_cohort
from steering_lite.benchmark.production import record_completed_stage, run_stages
from steering_lite.benchmark.sweep import MODEL_ID, dry_manifest


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
    parser.add_argument("--stage", choices=("bare", "vjp_cache"), default="vjp_cache")
    parser.add_argument("--backend", choices=("recorded",))
    parser.add_argument("--ledger", type=Path)
    args = parser.parse_args()
    if args.dry_run:
        result = dry_manifest(args.out, args.model)
    elif args.import_smoke:
        result = import_recorded_smoke(args.out, args.model)
    else:
        if args.backend != "recorded":
            raise SystemExit("--run requires a configured Modal backend; use --backend recorded only to test the completed smoke.")
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
