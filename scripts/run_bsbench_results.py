#!/usr/bin/env python3
"""Render the single BS-bench report artifact from a completed sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from steering_lite.benchmark.results import render_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("outputs/bsbench-v2"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    output = args.out or args.run_dir / "results"
    report = render_report(args.run_dir, output)
    print(json.dumps({
        "output": str(output),
        "points_sha256": report["artifact"]["points_sha256"],
        "point_count": len(report["artifact"]["points"]),
        "non_experimental": report["artifact"]["non_experimental"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
