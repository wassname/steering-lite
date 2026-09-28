"""Judged -C effect split by the bare answer's stance (PI/Claude, 2026-09-28).
Per model and method, at the -C Pareto-best dose (the dose that sets the score): mean judged premise change toward candour
(bare level minus steered level, Jev 0-8), split by the bare answer: accepts (level >= 6), middle, rejects (level <= 1).
Question: does VJP fail on the large models only where the bare model already rejects (no room), or also where it accepts?
Run: .venv/bin/python slop/reviews/2026-09-28_judged_by_stance/by_stance.py
"""
import json
from pathlib import Path
from statistics import mean
ROOT = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
MODELS = (("Qwen3.5-4B", "full"), ("Qwen3.5-27B", "27b-full"), ("OLMo-2-32B", "olmo-full"))
METHODS = ("mean_diff", "chars", "vjp_cache", "vjp_delta", "vjp_delta-t48", "vjp_delta-t47", "random")
bucket = lambda b: "accepts" if b >= 6 else ("rejects" if b <= 1 else "middle")
lines = ["| model | method | -C dose C | bare accepts: n, candour gain | middle: n, gain | bare rejects: n, gain |", "|---|---|---|---|---|---|"]
for model, res in MODELS:
    site = json.loads((ROOT / f"outputs/bsbench/results/{res}/points.json").read_text())
    best = {r["method"]: r["best"]["-C"] for r in site["summary"]}
    for m in METHODS:
        if m not in best or best[m] is None:
            continue
        pts = [p for p in site["points"] if p["method"] == m and p["side"] == "-C" and abs(p["C"] - best[m]["C"]) < 1e-9 and p["admissible"]]
        qs = [q for p in pts for q in p["questions"]]
        cells = []
        for b in ("accepts", "middle", "rejects"):
            g = [-q["effect"] for q in qs if bucket(q["bare_premise"]) == b]
            cells.append(f"{len(g)}, {mean(g):+.2f}" if g else "0, —")
        lines.append(f"| {model} | {m} | {best[m]['C']:.3g} | " + " | ".join(cells) + " |")
text = "\n".join(lines); print(text)
(here / "by_stance.md").write_text(__doc__ + "\n" + text + "\n")
