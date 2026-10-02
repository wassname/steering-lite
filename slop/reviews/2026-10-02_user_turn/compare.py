"""User-turn vs steering-everywhere, same vector (seed 0), same 100 questions. PI/OpenAI 2026-10-03.

Paired question bootstrap of the score difference (dose selection redone per draw, seed 0 only, so
seed-to-seed variation is NOT included). Jev audit (judge.audit_request) at each Pareto-best dose and on
bare answers; missing audit cells are rated (cost logged).

    PYTHONPATH=scripts/bsbench just --command uv run --extra benchmark python slop/reviews/2026-10-02_user_turn/compare.py
"""
import json
import math
import random
from pathlib import Path
from statistics import mean

from tabulate import tabulate

import judge
from results import directed, method_curve, pareto_score, random_curves, resample

OUT = Path("slop/reviews/2026-10-02_user_turn")
every = json.loads(Path("outputs/bsbench/results/full/points.json").read_text())
user = json.loads(Path("outputs/bsbench/results/user-full/points.json").read_text())
questions = {q["scenario"]: q for q in user["questions"]}
scenarios = list(questions)


def curves(points, method, seeds):
    group = [p for p in points if p["method"] == method and p["seed"] in seeds]
    if method == "random":
        return random_curves(group)
    return {side: method_curve(group, method, side) for side in ("+C", "-C")}


def audit_requests(point):
    return [judge.audit_request(questions[q["scenario"]]["prompt"], questions[q["scenario"]]["flaw"], q["text"]) for q in point["questions"]]


methods = sorted({p["method"] for p in user["points"]} & {p["method"] for p in every["points"]} - {"prompting", "prompting_engineered"})
rows, needed = [], {}
rng = random.Random(0)
best_points = {}
for method in methods:
    seeds = sorted({p["seed"] for p in every["points"] if p["method"] == "random"}) if method == "random" else [0]
    useeds = sorted({p["seed"] for p in user["points"] if p["method"] == "random"}) if method == "random" else [0]
    cu, ce = curves(user["points"], method, useeds), curves(every["points"], method, seeds)
    su, bu = pareto_score(cu)
    se, be = pareto_score(ce)
    diffs = []
    if method != "random":
        for _ in range(1000):
            drawn = [rng.choice(scenarios) for _ in scenarios]
            a = pareto_score({s: resample(c, drawn, [0]) for s, c in cu.items()})[0]
            b = pareto_score({s: resample(c, drawn, [0]) for s, c in ce.items()})[0]
            diffs.append((-math.inf if math.isnan(a) else a) - (-math.inf if math.isnan(b) else b))
        diffs.sort()
    best_points[method] = {"user": bu, "every": be}
    for view, best in (("user", bu), ("every", be)):
        for side, point in best.items():
            for request in audit_requests(point) if point else []:
                needed[judge.key(request)] = request
    rows.append({"method": method, "user": su, "every_s0": se, "diff": su - se if not (math.isnan(su) or math.isnan(se)) else float("nan"),
                 "diff_ci": (diffs[50], diffs[949]) if diffs else (float("nan"),) * 2,
                 "user_-C": bu["-C"] and (directed(bu["-C"]), bu["-C"]["off_axis"]), "every_-C": be["-C"] and (directed(be["-C"]), be["-C"]["off_axis"])})
bare = [judge.audit_request(q["prompt"], q["flaw"], q["bare"]) for q in user["questions"]]
for request in bare:
    needed[judge.key(request)] = request
missing = judge.refresh(needed, "audit_compare", True)
have = judge.cached()


def audit(requests):
    answers = [have[judge.key(r)] for r in requests]
    return mean(a["on_target"]["probabilities"]["yes"] for a in answers), mean(a["fabricates"]["probabilities"]["yes"] for a in answers)


bare_audit = audit(bare)
table = []
for row in sorted(rows, key=lambda r: -r["user"] if not math.isnan(r["user"]) else math.inf):
    cell = lambda view, side: "—" if best_points[row["method"]][view][side] is None else "{:.2f}/{:.2f}".format(*audit(audit_requests(best_points[row["method"]][view][side])))
    pair = lambda v: "—" if not v else f"{v[0]:+.2f}/{v[1]:.2f}"
    table.append([row["method"], row["user"], row["every_s0"], row["diff"], "—" if math.isnan(row["diff_ci"][0]) else f"[{row['diff_ci'][0]:+.2f}, {row['diff_ci'][1]:+.2f}]",
                  pair(row["user_-C"]), pair(row["every_-C"]), cell("user", "-C"), cell("every", "-C"), cell("user", "+C"), cell("every", "+C")])
headers = ["method", "score user", "score everywhere s0", "Δ", "Δ 90% CI (questions)", "−C on/off user", "−C on/off everywhere",
           "−C audit user", "−C audit everywhere", "+C audit user", "+C audit everywhere"]
text = tabulate(table, headers=headers, tablefmt="pipe", floatfmt="+.2f")
text += f"\n\naudit cells = P(on target)/P(fabricates), mean over answers at the Pareto-best dose. Bare answers: {bare_audit[0]:.2f}/{bare_audit[1]:.2f}. Random: user 16 directions vs everywhere 11 (unpaired, no Δ CI). Audit cells rated in this run: {missing}."
(OUT / "comparison.md").write_text(text + "\n")
print(text)
