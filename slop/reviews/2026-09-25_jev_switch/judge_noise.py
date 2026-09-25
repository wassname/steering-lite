"""Temporary check (PI/Claude, 2026-09-25): how much of the score uncertainty is the judge?
(1) repeat noise: re-ask Jev (rubric v2) on 300 answers, bypassing the cache.
(2) rubric-order noise: rate every answer at each method-side's score-setting dose (all seeds) plus the bare
    answers again with PREMISE and DAMAGE listed in reverse order; map back (level' = n-1 - level).
    Score at the same fixed doses: forward, reversed, and their mean (the 2-view ensemble).
(3) compare the spread between judge views with the question/seed bootstrap CI width.
    cd scripts/bsbench && ../../.venv/bin/python ../../slop/reviews/2026-09-25_jev_switch/judge_noise.py
"""
import asyncio, json, os, random, sys
from pathlib import Path
from statistics import mean
sys.path.insert(0, ".")
import httpx, numpy as np
import judge as J
import results as R
from data import COHORTS, default_model_dir, demo_rows, load_cohort, walk_certificates

OUT = Path("../../slop/reviews/2026-09-25_jev_switch")
REV = Path("../../outputs/bsbench/judgments/jev_reversed.jsonl")
model_dir = default_model_dir()

def reversed_request(q, flaw, a):
    r = J.aware_request(q, flaw, a)
    r["questions"]["premise"]["criteria"] = J.PREMISE[::-1]
    r["questions"]["damage"]["criteria"] = J.DAMAGE[::-1]
    return r

async def ask(requests, path=None):
    headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    have = {}
    if path and path.exists():
        have = {r["key"]: r["answers"] for r in map(json.loads, path.open())}
    todo = [r for r in requests if J.key(r) not in have]
    sem = asyncio.Semaphore(16)
    async def one(c, req):
        async with sem:
            for attempt in range(8):
                resp = await c.post(J.URL, json=req, headers=headers, timeout=60)
                if resp.status_code in (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529):
                    await asyncio.sleep(3 * 2**attempt); continue
                resp.raise_for_status(); return J.key(req), resp.json()["answers"]
            raise RuntimeError(resp.status_code)
    async with httpx.AsyncClient() as c:
        got = await asyncio.gather(*(one(c, r) for r in todo))
    if path:
        with path.open("a") as f:
            for k, a in got: f.write(json.dumps({"key": k, "answers": a}) + "\n")
    return have | dict(got)

points = R.build_points(model_dir, "full", set())
chosen = R.choose(points)
rows = {(r["method"], r["seed"], r["C"], r["side"], r["vignette"]): r for c in walk_certificates(model_dir, "full") for r in demo_rows(model_dir, c)}
fwd = J.cached()

# (1) repeat noise
rng = random.Random(0)
sample = rng.sample(list(rows.values()), 300)
reqs = [J.aware_request(r["prompt"], r["flaw"], r["steered"]) for r in sample]
again = asyncio.run(ask(reqs))
x = [fwd[J.key(q)]["premise"]["score"] for q in reqs]; y = [again[J.key(q)]["premise"]["score"] for q in reqs]
same = np.mean([abs(a - b) < 1e-9 for a, b in zip(x, y)])
print(f"## 1. Repeat noise\n\nRe-asked 300 answers (rubric v2): premise identical in {same:.0%}, r = {np.corrcoef(x, y)[0,1]:.3f}.\n")

# (2) reversed rubric at the score-setting doses (fixed doses: judge variation only)
targets = {}
for method, (_, best, _) in chosen.items():
    for side in ("+C", "-C"):
        p = best[side]
        for q in (p or {}).get("questions", []):
            r = rows[method, q["seed"], p["C"], side, q["scenario"]]
            targets[(method, side, q["seed"], q["scenario"])] = r
bares = {r["vignette"]: r for r in targets.values()}
reqs = [reversed_request(r["prompt"], r["flaw"], r["steered"]) for r in targets.values()] + [reversed_request(r["prompt"], r["flaw"], r["bare"]) for r in bares.values()]
rev = asyncio.run(ask(reqs, REV))
def levels(answers, reverse):
    n_p, n_d = len(J.PREMISE) - 1, len(J.DAMAGE) - 1
    p, d = answers["premise"]["score"], answers["damage"]["score"]
    return (n_p - p, n_d - d) if reverse else (p, d)
def view(r, which):
    out = []
    for text in (r["steered"], r["bare"]):
        f = levels(fwd[J.key(J.aware_request(r["prompt"], r["flaw"], text))], False)
        b = levels(rev[J.key(reversed_request(r["prompt"], r["flaw"], text))], True)
        out.append({"fwd": f, "rev": b, "ens": ((f[0] + b[0]) / 2, (f[1] + b[1]) / 2)}[which])
    (sp, sd), (bp, bd) = out
    return sp - bp, abs(sd - bd)
allf = [levels(fwd[J.key(J.aware_request(r["prompt"], r["flaw"], r["steered"]))], False)[0] for r in targets.values()]
allr = [levels(rev[J.key(reversed_request(r["prompt"], r["flaw"], r["steered"]))], True)[0] for r in targets.values()]
alldf = [levels(fwd[J.key(J.aware_request(r["prompt"], r["flaw"], r["steered"]))], False)[1] for r in targets.values()]
alldr = [levels(rev[J.key(reversed_request(r["prompt"], r["flaw"], r["steered"]))], True)[1] for r in targets.values()]
print(f"## 2. Rubric order (forward vs reversed level list), {len(targets)} steered answers at score-setting doses + {len(bares)} bare\n")
print(f"premise: r = {np.corrcoef(allf, allr)[0,1]:.3f}, mean shift reversed-forward {mean(allr) - mean(allf):+.2f} levels, mean |diff| {mean(abs(a-b) for a,b in zip(allf, allr)):.2f}")
print(f"damage:  r = {np.corrcoef(alldf, alldr)[0,1]:.3f}, mean shift {mean(alldr) - mean(alldf):+.2f}, mean |diff| {mean(abs(a-b) for a,b in zip(alldf, alldr)):.2f}\n")

# (3) score at fixed doses under each judge view vs the CI width
scen = list(load_cohort())[COHORTS["full"]]
print("## 3. Score at the fixed score-setting doses under each judge view, vs the question+seed CI\n")
print("| method | forward | reversed | 2-view mean | judge spread (max-min) | 90% CI width (seeds+questions) | judge share |\n|---|---|---|---|---|---|---|")
summary = {r["method"]: r for r in json.loads(Path("../../outputs/bsbench/results/full/points.json").read_text())["summary"]}
for method, (_, best, _) in chosen.items():
    if method in R.PROMPTS or any(best[s] is None for s in ("+C", "-C")):
        continue
    sc = {}
    for which in ("fwd", "rev", "ens"):
        per_side = []
        for side in ("+C", "-C"):
            vals = [view(r, which) for (m, s, _, _), r in targets.items() if m == method and s == side]
            on = mean(v[0] for v in vals) * (1 if side == "+C" else -1)
            per_side.append(on - R.OFF_WEIGHT * mean(v[1] for v in vals))
        sc[which] = min(per_side)
    lo, hi = summary[method]["ci"]
    spread = max(sc.values()) - min(sc.values())
    print(f"| {method} | {sc['fwd']:+.2f} | {sc['rev']:+.2f} | {sc['ens']:+.2f} | {spread:.2f} | {hi - lo:.2f} | {spread / (hi - lo):.0%} |")
