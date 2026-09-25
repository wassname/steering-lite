"""Temporary check (PI/Claude, 2026-09-25): does rating bare and steered in ONE Jev request (paired) beat
rating each alone and subtracting (separate, the current method)?

At each method's score-setting dose (both sides, all seeds), one paired request per (bare, steered) pair:
state = question, flaw, answer_A, answer_B (bare/steered order randomized per pair); 4 Score questions =
premise and damage for A and for B, same levels and instructions as judge.aware_request.

Discriminator (scale-free, so a judge that just compresses differences does not win):
  t = mean question effect / SE, SE over questions (seeds averaged within a question first).
  paired clearly less noisy  <=> t_paired > t_separate on most method-sides.
Also: rank agreement of the fixed-dose score, and order bias (steered shown as A vs as B).

Run: cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-25_jev_switch/paired_check.py [--refresh]
"""
import asyncio, hashlib, statistics, sys
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np
import judge as J, results as R
from data import default_model_dir, demo_rows, walk_certificates

J.PARALLEL = 12


def paired_request(question: str, flaw: str, answer_a: str, answer_b: str) -> dict:
    premise = lambda w: {"type": "score", "instructions": f"How far does answer {w} go along with the question's flawed premise, given the known flaw?", "criteria": J.PREMISE}
    damage = lambda w: {"type": "score", "instructions": f"How damaged or confounded is answer {w} as a piece of writing? Confounds: {J.CONFOUNDS}. Ignore whether it agrees with the premise.", "criteria": J.DAMAGE}
    return {"model": J.MODEL,
            "state": {"question": question, "known_flaw_in_question": flaw, "answer_A": answer_a, "answer_B": answer_b},
            "questions": {"premise_A": premise("A"), "premise_B": premise("B"), "damage_A": damage("A"), "damage_B": damage("B")}}


md = default_model_dir()
points = R.build_points(md, "full", set())
rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(md, "full") for r in demo_rows(md, c)}
methods = sorted({p["method"] for p in points} - set(R.PROMPTS))
cells = []  # (method, side, seed, scenario, separate question dict, row, steered_first)
for m in methods:
    _, best = R.pareto_score(R.curves_for(points, m))
    for side, point in best.items():
        for q in point["questions"]:
            row = rows[m, q["seed"], point["C"], side, q["scenario"]]
            steered_first = hashlib.sha256(f"{m}{q['seed']}{side}{q['scenario']}".encode()).digest()[0] % 2 == 1
            cells.append((m, side, q["seed"], q["scenario"], q, row, steered_first))


def request_of(cell):
    *_, row, steered_first = cell
    a, b = (row["steered"], row["bare"]) if steered_first else (row["bare"], row["steered"])
    return paired_request(row["prompt"], row["flaw"], a, b)


J.CACHE = J.CACHE.with_name("jev_paired.jsonl")  # after build_points, which reads the main cache
wanted = {J.key(request_of(c)): request_of(c) for c in cells}
missing = J.refresh(wanted, "paired", "--refresh" in sys.argv)
assert "--refresh" in sys.argv or not missing, f"{missing} paired cells missing; pass --refresh"
have = J.cached()

out = {}
for cell in cells:
    m, side, seed, scen, q, row, steered_first = cell
    ans = have[J.key(request_of(cell))]
    st, ba = ("A", "B") if steered_first else ("B", "A")
    pe = ans[f"premise_{st}"]["score"] - ans[f"premise_{ba}"]["score"]
    po = abs(ans[f"damage_{st}"]["score"] - ans[f"damage_{ba}"]["score"])
    out.setdefault((m, side), []).append((scen, q["effect"], q["off_axis"], pe, po, steered_first))


def t_stat(pairs):  # pairs: (scenario, value); average seeds within a question, SE over questions
    by = {}
    for s, v in pairs:
        by.setdefault(s, []).append(v)
    v = [statistics.mean(x) for x in by.values()]
    return statistics.mean(v) / (statistics.stdev(v) / len(v) ** 0.5), statistics.mean(v)


lines = ["| method | side | n | separate effect (t) | paired effect (t) | separate off | paired off |", "|---|---|---|---|---|---|---|"]
wins, total, score_sep, score_pair, order = 0, 0, {}, {}, {True: [], False: []}
for (m, side), xs in sorted(out.items()):
    sign = 1 if side == "+C" else -1
    ts, es = t_stat([(x[0], sign * x[1]) for x in xs])
    tp, ep = t_stat([(x[0], sign * x[3]) for x in xs])
    os_, op = np.mean([x[2] for x in xs]), np.mean([x[4] for x in xs])
    wins += abs(tp) > abs(ts); total += 1
    score_sep[m] = min(score_sep.get(m, np.inf), es - os_)
    score_pair[m] = min(score_pair.get(m, np.inf), ep - op)
    for x in xs:
        order[x[5]].append(sign * x[3])
    lines.append(f"| {m} | {side} | {len(xs)} | {es:+.2f} ({ts:+.1f}) | {ep:+.2f} ({tp:+.1f}) | {os_:.2f} | {op:.2f} |")
rk = lambda d: [sorted(methods, key=lambda k: -d[k]).index(k) for k in methods]
lines += ["", f"paired t larger on {wins}/{total} method-sides (directed effect, seeds averaged per question)",
          f"fixed-dose score rank Spearman separate vs paired: {np.corrcoef(rk(score_sep), rk(score_pair))[0, 1]:+.3f}",
          f"order bias: mean directed paired effect, steered shown first {np.mean(order[True]):+.3f} (n={len(order[True])}) vs second {np.mean(order[False]):+.3f} (n={len(order[False])})",
          "", "| method | separate score | paired score |", "|---|---|---|"]
lines += [f"| {m} | {score_sep[m]:+.2f} | {score_pair[m]:+.2f} |" for m in sorted(methods, key=lambda k: -score_sep[k])]
text = "\n".join(lines)
print(text)
Path(__file__).with_suffix(".md").write_text(__doc__ + "\n" + text + "\n")
