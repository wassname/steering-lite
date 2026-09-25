"""Where does each method's CI width come from: seeds or questions? Decides where extra seeds would help.

90% CI width of the score (min over ±C of on - off, dose selection redone per draw) when resampling
seeds only (all questions), questions only (all seeds), or both (the results.py hierarchical bootstrap).
Rule (goal 4): add seeds where the seeds-only width is more than 25% of the both width.

    python ci_decomposition.py --cohort full > ../../outputs/logs/ci-decomposition-full.txt
"""

import argparse
import math
import random

from judge import COHORTS, default_model_dir, load_cohort
from results import PROMPTS, build_points, method_curve, pareto_score, random_curves, resample

N_DRAWS = 400
SEED_SHARE_LIMIT = 0.25


def width(curves: dict, seeds: list[int], scenarios: list[str], vary_seeds: bool, vary_questions: bool, rng: random.Random) -> float:
    scores = []
    for _ in range(N_DRAWS):
        drawn_seeds = [rng.choice(seeds) for _ in seeds] if vary_seeds else seeds
        drawn = [rng.choice(scenarios) for _ in scenarios] if vary_questions else scenarios
        score, _ = pareto_score({side: resample(curve, drawn, drawn_seeds) for side, curve in curves.items()})
        scores.append(-math.inf if math.isnan(score) else score)
    scores.sort()
    return scores[int(0.95 * N_DRAWS) - 1] - scores[int(0.05 * N_DRAWS)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=tuple(COHORTS), default="full")
    args = parser.parse_args()
    points = build_points(default_model_dir(), args.cohort, set())
    scenarios = list(load_cohort())[COHORTS[args.cohort]]
    print(f"| method | seeds | 90% CI width: seeds only | questions only | both | seeds-only / both | add seeds (> {SEED_SHARE_LIMIT:.0%}) |")
    print("|---|--:|--:|--:|--:|--:|---|")
    for method in sorted({p["method"] for p in points} - set(PROMPTS), key=lambda m: (m == "random", m)):
        curves = random_curves(points) if method == "random" else {side: method_curve(points, method, side) for side in ("+C", "-C")}
        seeds = sorted({p["seed"] for p in points if p["method"] == method})
        rng = random.Random(method)
        w_seed, w_q, w_both = (width(curves, seeds, scenarios, s, q, rng) for s, q in ((True, False), (False, True), (True, True)))
        share = w_seed / w_both
        print(f"| {method} | {len(seeds)} | {w_seed:.2f} | {w_q:.2f} | {w_both:.2f} | {share:.0%} | {'YES' if share > SEED_SHARE_LIMIT else 'no'} |")


if __name__ == "__main__":
    main()
