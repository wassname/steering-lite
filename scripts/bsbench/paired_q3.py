"""Paired bootstrap of the dev score difference between two methods: the same drawn questions for both (they share
the 20 questions and the judge), seeds drawn per method; dose selection redone in each draw, as in results.py.

    python paired_q3.py sink_value mean_diff
"""
import math
import random
import sys
from pathlib import Path

from data import COHORTS, load_cohort
from results import build_points, curves_for, pareto_score, resample

a, b = sys.argv[1:3]
model_dir = Path(__file__).resolve().parents[2] / "outputs/bsbench/Qwen--Qwen3-4B-g7c7712c6"
points = build_points(model_dir, "dev", set())
scenarios = list(load_cohort())[COHORTS["dev"]]
curves = {m: curves_for(points, m) for m in (a, b)}
seeds = {m: sorted({p["seed"] for p in points if p["method"] == m}) for m in (a, b)}
rng, diffs = random.Random(0), []
for _ in range(2000):
    drawn = [rng.choice(scenarios) for _ in scenarios]
    s = {}
    for m in (a, b):
        ds = [rng.choice(seeds[m]) for _ in seeds[m]]
        score, _ = pareto_score({side: resample(c, drawn, ds) for side, c in curves[m].items()})
        s[m] = -math.inf if math.isnan(score) else score
    diffs.append(s[a] - s[b])
diffs.sort()
point = pareto_score(curves[a])[0] - pareto_score(curves[b])[0]
print(f"score({a}) - score({b}) = {point:+.3f}; paired 90% CI [{diffs[100]:+.3f}, {diffs[1899]:+.3f}]; "
      f"P(diff > 0) = {sum(d > 0 for d in diffs) / len(diffs):.2f}; seeds {seeds[a]} vs {seeds[b]}, n_questions={len(scenarios)}")
