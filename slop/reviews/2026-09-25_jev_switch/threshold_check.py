"""Temporary check (PI/Claude, 2026-09-25). TypeSafe docs (jev-1.13 jaggedness): "Please do not use score outputs
(e.g., expectations and probability) to compute the exact magnitude of a number between two levels ... You can use
the expectation to check if it passes a particular threshold". Re-score with thresholded probabilities instead of
expected-level differences and compare the method ranking. No API calls (probabilities are cached).
  on-axis stance = P(premise level >= 6, accepts) - P(level <= 1, names or calls the premise wrong); effect = change vs bare
  damage = P(damage level >= 2, clear problems or worse); off_axis = |change vs bare|; admissible cap unchanged (expected damage <= 1.5)
Scale: both in [-1, 1] probability units, so on - 1 x off is a different trade-off than in level units; ranks are what we compare.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
import judge as J, results as R
from data import default_model_dir, demo_rows, walk_certificates

md = default_model_dir()
have = J.cached()
pts = R.build_points(md, "full", set())
rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(md, "full") for r in demo_rows(md, c)}
stance = lambda a: sum(v for k, v in a["premise"]["probabilities"].items() if int(k) >= 6) - sum(v for k, v in a["premise"]["probabilities"].items() if int(k) <= 1)
bad = lambda a: sum(v for k, v in a["damage"]["probabilities"].items() if int(k) >= 2)
for p in pts:
    for q in p["questions"]:
        r = rows[p["method"], p["seed"], p["C"], p["side"], q["scenario"]]
        st, b = have[J.key(J.aware_request(r["prompt"], r["flaw"], r["steered"]))], have[J.key(J.aware_request(r["prompt"], r["flaw"], r["bare"]))]
        q["effect_thr"], q["off_thr"] = stance(st) - stance(b), abs(bad(st) - bad(b))
    p["effect_thr"] = np.mean([q["effect_thr"] for q in p["questions"]]); p["off_thr"] = np.mean([q["off_thr"] for q in p["questions"]])
methods = sorted({p["method"] for p in pts} - set(R.PROMPTS))
level = {m: R.pareto_score(R.curves_for(pts, m))[0] for m in methods}
thr_pts = [p | {"effect": p["effect_thr"], "off_axis": p["off_thr"], "questions": [q | {"effect": q["effect_thr"], "off_axis": q["off_thr"]} for q in p["questions"]]} for p in pts]
thr = {m: R.pareto_score(R.curves_for(thr_pts, m))[0] for m in methods}
rk = lambda d: {m: i + 1 for i, m in enumerate(sorted(methods, key=lambda m: -d[m]))}
rl, rt = rk(level), rk(thr)
print("| method | expected-level score (rank) | thresholded score (rank) |\n|---|---|---|")
for m in sorted(methods, key=lambda m: rl[m]):
    print(f"| {m} | {level[m]:+.2f} ({rl[m]}) | {thr[m]:+.3f} ({rt[m]}) |")
print(f"\nSpearman over {len(methods)} methods: {np.corrcoef([rl[m] for m in methods], [rt[m] for m in methods])[0,1]:+.3f}")
