"""$0 checks requested by the review: (1) per-seed everywhere scores, is seed 0 typical? (2) split-half dose
selection: pick each side's dose on 50 questions, score it on the other 50, for both views. PI/OpenAI 2026-10-03.

    PYTHONPATH=scripts/bsbench uv run --extra benchmark python slop/reviews/2026-10-02_user_turn/split_half.py
"""
import json
import math
import random
from pathlib import Path
from statistics import mean, median

from results import OFF_WEIGHT, directed, method_curve, pareto_score, resample

every = json.loads(Path("outputs/bsbench/results/full/points.json").read_text())
user = json.loads(Path("outputs/bsbench/results/user-full/points.json").read_text())
scenarios = [q["scenario"] for q in user["questions"]]
METHODS = ["vjp_resid", "sspace_scale", "vjp_value", "corda_pca", "mean_diff", "linear_act", "chars"]


def curves(points, method, seed):
    group = [p for p in points if p["method"] == method and p["seed"] == seed]
    return {side: method_curve(group, method, side) for side in ("+C", "-C")}


print("## Everywhere, one seed at a time (same rule as the report)\n")
print("| method | s0 | s1 | s2 | user s0 |\n|---|---:|---:|---:|---:|")
for m in METHODS:
    cells = [pareto_score(curves(every["points"], m, s))[0] for s in (0, 1, 2)]
    print(f"| {m} | " + " | ".join(f"{c:+.2f}" for c in cells) + f" | {pareto_score(curves(user['points'], m, 0))[0]:+.2f} |")


def held_out(side_curves, pick, score):
    """Choose each side's dose on `pick` questions, report on-axis − off-axis on `score` questions; min over sides."""
    out = []
    for side, curve in side_curves.items():
        chosen = resample(curve, pick, [0])
        if not chosen:
            return -math.inf
        best = max(chosen, key=lambda p: directed(p) - OFF_WEIGHT * p["off_axis"])
        held = resample([p for p in curve if p["C"] == best["C"]], score, [0])
        if not held:  # damage cap fails on the held-out half
            return -math.inf
        out.append(directed(held[0]) - OFF_WEIGHT * held[0]["off_axis"])
    return min(out)


rng = random.Random(0)
splits = []
for _ in range(200):
    shuffled = scenarios[:]
    rng.shuffle(shuffled)
    splits.append((shuffled[:50], shuffled[50:]))
print("\n## Split-half dose selection, 200 random 50/50 splits, seed 0\n")
print("| method | user held-out median | everywhere held-out median | Δ median | Δ 5–95% over splits | in-sample Δ |\n|---|---:|---:|---:|---|---:|")
for m in METHODS:
    cu, ce = curves(user["points"], m, 0), curves(every["points"], m, 0)
    u = [held_out(cu, a, b) for a, b in splits]
    e = [held_out(ce, a, b) for a, b in splits]
    d = sorted(x - y for x, y in zip(u, e))
    full = pareto_score(cu)[0] - pareto_score(ce)[0]
    print(f"| {m} | {median(u):+.2f} | {median(e):+.2f} | {median(d):+.2f} | [{d[10]:+.2f}, {d[189]:+.2f}] | {full:+.2f} |")
print("\nSplit spread reflects which questions pick vs score the dose; it is not a confidence interval and omits seed variation.")
