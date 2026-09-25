"""Temporary check (PI/Claude, 2026-09-25) for the oracle review: (1) ranking under other off-axis weights and
damage caps, (2) paired method - random contrast on shared question draws, (3) rank vs the frozen DeepSeek table.
    cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-25_jev_switch/robustness.py
"""
import math, random, re, sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import results as R
from data import COHORTS, default_model_dir, load_cohort

points = R.build_points(default_model_dir(), "full", set())
scenarios = list(load_cohort())[COHORTS["full"]]
methods = sorted({p["method"] for p in points} - set(R.PROMPTS))
seeds = {m: sorted({p["seed"] for p in points if p["method"] == m}) for m in methods}

def scores(weight, cap):
    R.OFF_WEIGHT, R.MAX_DAMAGE = weight, cap
    pts = [p | {"admissible": not p["breakdown_reasons"] and not p["post_boundary"] and p["steered_damage"] <= cap} for p in points]
    return {m: R.pareto_score(R.curves_for(pts, m))[0] for m in methods}, pts

print("## 1. Top 5 under other off-axis weights and damage caps\n")
print("| weight | cap | top 5 (score) | random rank |\n|---|---|---|---|")
for w in (0.5, 1.0, 2.0):
    for cap in (1.0, 1.5, 2.0):
        s, _ = scores(w, cap)
        order = sorted(methods, key=lambda m: -np.nan_to_num(s[m], nan=-9))
        print(f"| {w} | {cap} | " + ", ".join(f"{m} {s[m]:+.2f}" for m in order[:5]) + f" | {order.index('random') + 1} of {len(methods)} |")

s, pts = scores(1.0, 1.5)
curves = {m: R.curves_for(pts, m) for m in methods}
print("\n## 2. Paired contrast vs random (weight 1, cap 1.5; 400 draws; same question draw for both, seeds drawn per method)\n")
print("| method | score - random | 90% CI | P(method > random) |\n|---|---|---|---|")
rows = []
for m in [m for m in methods if m != "random"]:
    diffs = []
    for d in range(400):
        rng = random.Random(d)
        drawn = [rng.choice(scenarios) for _ in scenarios]
        vals = []
        for k in (m, "random"):
            ds = [rng.choice(seeds[k]) for _ in seeds[k]]
            v, _ = R.pareto_score({side: R.resample(c, drawn, ds) for side, c in curves[k].items()})
            vals.append(-math.inf if math.isnan(v) else v)
        diffs.append(vals[0] - vals[1])
    diffs.sort()
    rows.append((s[m] - s["random"], m, diffs[20], diffs[379], np.mean([x > 0 for x in diffs])))
for point, m, lo, hi, p in sorted(rows, reverse=True):
    print(f"| {m} | {point:+.2f} | [{lo:+.2f}, {hi:+.2f}] | {p:.0%} |")

print("\n## 3. Rank vs the DeepSeek judge (frozen table, same walks)\n")
ds = {}
for line in Path("../../slop/reviews/2026-09-25_deepseek_vs_jev/index_deepseek.md").read_text().splitlines():
    m = re.match(r"\| \*?([a-z_]+)\*? \| ([+-]\d\.\d\d) \|", line)
    if m and m.group(1) in methods: ds[m.group(1)] = float(m.group(2))
common = [m for m in methods if m in ds]
rk = lambda v: np.argsort(np.argsort(v))
r = np.corrcoef(rk([ds[m] for m in common]), rk([s[m] for m in common]))[0, 1]
print(f"Spearman over {len(common)} methods (incl. random): {r:+.2f}. DeepSeek top 5: {sorted(common, key=lambda m: -ds[m])[:5]}; Jev v2 top 5: {sorted(common, key=lambda m: -s[m])[:5]}")
