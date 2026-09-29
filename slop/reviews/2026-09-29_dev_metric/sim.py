"""Offline simulation of a cheap dev metric on the cached Qwen3.5-4B full walks (PI/Claude, 2026-09-29).

No GPU, no judge: every answer and Jev rating is already cached. Question: does a cheap dev score rank methods like
the full headline score (outputs/bsbench/results/full/points.json, 100 q, 3 seeds, best admissible dose)?

Dev rule (wassname 2026-09-28): find breakdown with the health check only, step back to 0.67 x C_break, judge that one
dose per side. score = min over sides of (on - off) at that dose, seed 0.
  cap128: health recomputed on answers cut to 128 tokens (a cut answer counts as unfinished).
Question sets: every 5th (current dev), 20 random, 20 "informative" (largest spread across methods, chosen on half the
methods, scored on the other half), all 100.
Run: .venv/bin/python slop/reviews/2026-09-29_dev_metric/sim.py > slop/reviews/2026-09-29_dev_metric/sim.md
"""
import json, random, sys
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer

def spearmanr(x, y):
    rank = lambda v: {i: r for r, i in enumerate(sorted(range(len(v)), key=lambda i: v[i]))}
    rx, ry = rank(x), rank(y); n = len(x)
    return (1 - 6 * sum((rx[i] - ry[i]) ** 2 for i in range(n)) / (n * (n * n - 1))),


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/bsbench"))
import walk  # noqa: E402  health() is the walk's own breakdown check

MD = ROOT / "outputs/bsbench/Qwen--Qwen3.5-4B-g7c7712c6"
site = json.loads((ROOT / "outputs/bsbench/results/full/points.json").read_text())
full = {r["method"]: r["score"] for r in site["summary"] if r["score"] is not None and not r["method"].startswith("prompting")}
methods = sorted(full)
pts = {(p["method"], p["seed"], round(p["C"], 6), p["side"]): p for p in site["points"] if p["seed"] == 0}
del site
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
scen = [q for q in json.loads(json.dumps(walk.read_cohort("full")))]
scenarios = [r["scenario"] for r in scen]


def capped(texts, cap=128):
    ids = tok(texts).input_ids
    return [t if len(i) <= cap else tok.decode(i[:cap]) + " …" for t, i in zip(texts, ids)]


def dev_dose(method, side, cap):
    """C of the rung nearest below 0.67 x first-unhealthy C (seed 0); also returns the first-unhealthy C."""
    cert = json.loads((MD / f"walks/{method}_s0_full.json").read_text())
    rungs = sorted(cert["rungs"], key=lambda r: r["coefficient"])
    brk = None
    for r in rungs:
        if cap:
            texts = [json.loads(l)["text"] for l in open(MD / r[side]["answers"])]
            broken = bool(walk.health(tok, capped(texts))[1])
        else:
            broken = bool(r[side]["breakdown_reasons"])
        if broken:
            brk = r["coefficient"]; break
    assert brk is not None, (method, side)
    below = [r["coefficient"] for r in rungs if r["coefficient"] <= 0.67 * brk]
    return (below[-1] if below else rungs[0]["coefficient"]), brk


def per_q(method, side, C):
    p = pts[(method, 0, round(C, 6), side)]
    sign = 1 if side == "+C" else -1
    return {q["scenario"]: sign * q["effect"] - q["off_axis"] for q in p["questions"]}


def score(table, method, qs):
    return min(mean(table[method][side][s] for s in qs) for side in ("+C", "-C"))


out = []
for cap in (False, True):
    doses, table, moved = {}, {}, 0
    for m in methods:
        table[m] = {}
        for side in ("+C", "-C"):
            C, brk = dev_dose(m, side, cap)
            doses[m, side] = (C, brk)
            table[m][side] = per_q(m, side, C)
    if cap:
        moved = sum(doses[k][1] != base_doses[k][1] for k in doses)
    else:
        base_doses = dict(doses)
    every5 = scenarios[::5]
    rng = random.Random(0)
    rows = {"all 100": scenarios, "every 5th (current dev)": every5}
    rand = [spearmanr([full[m] for m in methods], [score(table, m, rng.sample(scenarios, 20)) for m in methods])[0] for _ in range(200)]
    # informative: choose on half A (alternating by full rank), score on half B, and swap
    ranked = sorted(methods, key=lambda m: -full[m])
    halves = (ranked[0::2], ranked[1::2])
    info_rho = []
    for a, b in (halves, halves[::-1]):
        spread = {s: sum((table[m][side][s] - mean(table[x][side][s] for x in a)) ** 2 for m in a for side in ("+C", "-C")) for s in scenarios}
        pick = sorted(scenarios, key=lambda s: -spread[s])[:20]
        info_rho.append(spearmanr([full[m] for m in b], [score(table, m, pick) for m in b])[0])
        info_rho.append(spearmanr([full[m] for m in b], [score(table, m, every5) for m in b])[0])
    out.append(f"\n## health {'on answers cut to 128 tokens' if cap else 'as walked (512 tokens)'}"
               + (f": breakdown rung moved on {moved} of {len(doses)} method-sides vs 512" if cap else ""))
    out.append("| question set (20 unless noted) | Spearman vs full score, over %d methods |" % len(methods))
    out.append("|---|---|")
    for name, qs in rows.items():
        out.append(f"| {name} | {spearmanr([full[m] for m in methods], [score(table, m, qs) for m in methods])[0]:+.2f} |")
    rs = sorted(rand)
    out.append(f"| 20 random (200 draws): median [5%, 95%] | {rs[100]:+.2f} [{rs[10]:+.2f}, {rs[190]:+.2f}] |")
    out.append(f"| informative, held-out half (2 splits) vs every 5th on the same half | {info_rho[0]:+.2f}, {info_rho[2]:+.2f} vs {info_rho[1]:+.2f}, {info_rho[3]:+.2f} |")
    if not cap:
        out.append("\n| method | full score | dev dose -C / C_break | dev dose +C / C_break | dev score, every 5th |")
        out.append("|---|---|---|---|---|")
        for m in ranked:
            out.append(f"| {m} | {full[m]:+.2f} | {doses[m,'-C'][0]:.3g} / {doses[m,'-C'][1]:.3g} | {doses[m,'+C'][0]:.3g} / {doses[m,'+C'][1]:.3g} | {score(table, m, every5):+.2f} |")
print(__doc__ + "\n".join(out))


# --- follow-up: step-back factor, and the current dev protocol as baseline (PI/Claude) ---
def best_admissible(method, side, qs):
    """Current protocol on a question subset, seed 0: max over doses of (on - off) among doses with mean damage <= 1.5."""
    sign = 1 if side == "+C" else -1
    best = None
    for (m, seed, C, s), p in pts.items():
        if m != method or s != side or p["breakdown_reasons"]:
            continue
        qq = [q for q in p["questions"] if q["scenario"] in qs]
        if mean(q["steered_damage"] for q in qq) > 1.5:
            continue
        v = mean(sign * q["effect"] - q["off_axis"] for q in qq)
        best = v if best is None else max(best, v)
    return best if best is not None else float("-inf")


brk = {(m, side): base_doses[m, side][1] for m in methods for side in ("+C", "-C")}
lines = ["\n## Step-back factor (health as walked) and the current dev protocol",
         "| rule | all 100 | every 5th | 20 random, median [5%, 95%] |", "|---|---|---|---|"]
for factor in (0.67, 0.5, 0.4, 0.3, 0.2):
    t = {}
    for m in methods:
        t[m] = {}
        for side in ("+C", "-C"):
            rungs = sorted({C for (mm, _, C, s) in pts if mm == m and s == side})
            below = [C for C in rungs if C <= factor * brk[m, side] + 1e-9]
            t[m][side] = per_q(m, side, below[-1] if below else rungs[0])
    r = sorted(spearmanr([full[m] for m in methods], [score(t, m, random.Random(i).sample(scenarios, 20)) for m in methods])[0] for i in range(200))
    lines.append(f"| one dose at {factor} x C_break | {spearmanr([full[m] for m in methods], [score(t, m, scenarios) for m in methods])[0]:+.2f} | "
                 f"{spearmanr([full[m] for m in methods], [score(t, m, scenarios[::5]) for m in methods])[0]:+.2f} | {r[100]:+.2f} [{r[10]:+.2f}, {r[190]:+.2f}] |")
ba = lambda qs: [min(best_admissible(m, s, set(qs)) for s in ("+C", "-C")) for m in methods]
r = sorted(spearmanr([full[m] for m in methods], ba(random.Random(i).sample(scenarios, 20)))[0] for i in range(50))
lines.append(f"| current: walk + best judged dose (seed 0) | {spearmanr([full[m] for m in methods], ba(scenarios))[0]:+.2f} | "
             f"{spearmanr([full[m] for m in methods], ba(scenarios[::5]))[0]:+.2f} | {r[25]:+.2f} [{r[2]:+.2f}, {r[47]:+.2f}] (50 draws) |")
print("\n".join(lines))
