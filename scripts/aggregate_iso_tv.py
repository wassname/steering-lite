"""Aggregate iso-TV results across (model, method, seed) into a reliability table.

Reliability metrics:
- SI mean ± std across seeds: how consistent
- |mean(SI)| / std(SI): "surgical reliability ratio" (SR), higher = more reliable
- coeff variability across seeds: how stable the calibration is
- TV achieved: did the calibration converge near target?

Outputs a markdown table to stdout.
"""
from __future__ import annotations
import json
import sys
import statistics as stats
from pathlib import Path

OUT = Path("outputs/iso_tv")
TARGET_TV = 0.05

models = {
    "Qwen3-0.6B": "Qwen--Qwen3-0.6B-Base",
    "Gemma-3-1B": "google--gemma-3-1b-it",
    "Qwen3-4B": "Qwen--Qwen3-4B-Base",
    "Gemma-3-4B": "google--gemma-3-4b-it",
}

methods = ["mean_diff", "pca", "topk_clusters", "cosine_gated", "sspace", "spherical"]


def load_latest(model_tag: str):
    candidates = sorted(OUT.glob(f"iso_tv__{model_tag}__L*__seeds0_1_2__*.json"))
    if not candidates:
        return None
    return json.loads(candidates[-1].read_text())


def agg(values):
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return stats.mean(values), stats.stdev(values)


print("# Iso-TV reliability table\n")
print("Per (model, method): coeff*, TV achieved, SI, SI_JS, leakage, target_effect — mean±std over 3 seeds. SR = |mean(SI)| / std(SI). Higher SR = more reliable across seeds.\n")

for short, tag in models.items():
    data = load_latest(tag)
    if data is None:
        print(f"## {short} -- NO DATA\n")
        continue
    summary = data["summary"]
    layers = data["args"]["layers"]
    print(f"## {short} (L{layers})\n")
    print("| method | coeff* | TV* | target | leakage | SI | SR | SI_JS |")
    print("|---|---|---|---|---|---|---:|---|")
    rows = []
    for m in methods:
        rs = [r for r in summary if r["method"] == m]
        coeffs = [r["calibrated_coeff"] for r in rs]
        tvs = [r["tv_target"] for r in rs]
        sis = [r["si"] for r in rs]
        leaks = [r["leakage"] for r in rs]
        tgts = [r["target_effect"] for r in rs]
        sjs = [r["si_js"] for r in rs]
        c_m, c_s = agg(coeffs)
        tv_m, tv_s = agg(tvs)
        si_m, si_s = agg(sis)
        leak_m, leak_s = agg(leaks)
        tgt_m, tgt_s = agg(tgts)
        sj_m, sj_s = agg(sjs)
        sr = abs(si_m) / si_s if si_s > 0 else float("inf")
        rows.append((m, c_m, c_s, tv_m, tv_s, tgt_m, tgt_s, leak_m, leak_s, si_m, si_s, sr, sj_m, sj_s))
    rows.sort(key=lambda r: -r[11])  # sort by SR desc
    for (m, c_m, c_s, tv_m, tv_s, tgt_m, tgt_s, leak_m, leak_s, si_m, si_s, sr, sj_m, sj_s) in rows:
        print(
            f"| {m} | {c_m:.3g}±{c_s:.2g} | {tv_m:.3f}±{tv_s:.3f} | "
            f"{tgt_m:+.3f}±{tgt_s:.3f} | {leak_m:+.3f}±{leak_s:.3f} | "
            f"{si_m:+.3f}±{si_s:.3f} | {sr:.2f} | {sj_m:+.5f}±{sj_s:.5f} |"
        )
    print()

print("\n## Cross-model summary (mean SR per method, higher = more reliable)\n")
print("| method | Qwen3-0.6B | Gemma-3-1B | Qwen3-4B | Gemma-3-4B | mean SR |")
print("|---|---:|---:|---:|---:|---:|")
for m in methods:
    sr_per_model = []
    for short, tag in models.items():
        data = load_latest(tag)
        if data is None:
            sr_per_model.append(float("nan"))
            continue
        rs = [r for r in data["summary"] if r["method"] == m]
        sis = [r["si"] for r in rs]
        if len(sis) < 2:
            sr_per_model.append(float("nan"))
            continue
        si_m, si_s = stats.mean(sis), stats.stdev(sis)
        sr_per_model.append(abs(si_m) / si_s if si_s > 0 else float("inf"))
    valid = [x for x in sr_per_model if x == x and x != float("inf")]
    mean_sr = stats.mean(valid) if valid else float("nan")
    cells = " | ".join(f"{x:.2f}" if x == x else "n/a" for x in sr_per_model)
    print(f"| {m} | {cells} | {mean_sr:.2f} |")
