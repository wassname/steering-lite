"""Write the non-paying BS-bench sweep manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_lite.benchmark.sweep import MODEL_ID, dry_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--out", type=Path, default=Path("outputs/bsbench-v2"))
    args = parser.parse_args()
    if not args.dry_run:
        raise SystemExit("This entrypoint is a non-paid preflight; paid execution is not enabled.")
    manifest = dry_manifest(args.out, args.model)
    reused = sum(stage["reused"] for stage in manifest["stages"])
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "stages": len(manifest["stages"]), "reused": reused, "cost_estimate": manifest["cost_estimate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
