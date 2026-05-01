"""Summarize latest AIRisk benches into simple nats tables."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _latest_json(dirpath: Path) -> Path:
    files = sorted(p for p in dirpath.glob("*.json") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no json outputs in {dirpath}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _latest_csv(dirpath: Path) -> Path:
    files = sorted(p for p in dirpath.glob("*__per_row.csv") if p.is_file())
    if not files:
        raise FileNotFoundError(f"no per-row csv outputs in {dirpath}")
    return max(files, key=lambda p: p.stat().st_mtime)


def _read_rows(path: Path, target: str) -> dict[str, float]:
    out = {}
    key = f"logratio_{target}"
    for row in csv.DictReader(open(path)):
        v = row[key]
        if v not in ("", "nan"):
            out[row["idx"]] = float(v)
    return out


def _bootstrap_ci(deltas: np.ndarray, *, seed: int = 0, rounds: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = []
    n = len(deltas)
    for _ in range(rounds):
        sample = deltas[rng.integers(0, n, size=n)]
        boots.append(sample.mean())
    lo, hi = np.quantile(np.array(boots), [0.025, 0.975])
    return float(lo), float(hi)


def _compare(base_dir: Path, run_dir: Path, target: str, label: str) -> dict:
    base_rows = _read_rows(_latest_csv(base_dir), target)
    run_rows = _read_rows(_latest_csv(run_dir), target)
    common = sorted(set(base_rows) & set(run_rows))
    deltas = np.array([run_rows[i] - base_rows[i] for i in common], dtype=float)
    js = json.loads(_latest_json(run_dir).read_text())
    lo, hi = _bootstrap_ci(deltas)
    return {
        "label": label,
        "delta_mean": float(deltas.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(len(common)),
        "pmass": float(js["summary"]["steered_pmass_mean"]),
        "think_tokens": float(js["summary"]["steered_think_tokens_mean"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True, help="run_airisk_readme_latest full/ dir")
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    base_dir = args.dir
    rows = []
    for sub in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        rows.append(_compare(base_dir, sub, args.target, sub.name))
    rows.sort(key=lambda r: r["delta_mean"], reverse=True)
    print(json.dumps({"method_rows": rows}, indent=2))


if __name__ == "__main__":
    main()
