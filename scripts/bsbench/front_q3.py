"""-C (less sycophantic) Pareto front: best premise-rejection shift reached at each damage cap, per method, Qwen3-4B dev.

The +C side sits on a noise floor on Qwen3-4B (bare already accepts most premises), so the -C front separates methods.
For random, each seed is a row and the table also gives the max over seeds (the random zone's edge).
"""
import json
import sys
from pathlib import Path

from tabulate import tabulate

pts = json.load(open(Path(sys.argv[1]) / "points.json"))["points"]
CAPS = (0.4, 0.6, 0.8, 1.0)
rows = []
for m in sorted({p["method"] for p in pts}):
    for seed in sorted({p["seed"] for p in pts if p["method"] == m}):
        r = [p for p in pts if p["method"] == m and p["seed"] == seed and p["side"] == "-C" and p["admissible"]]
        row = {"method": m if m != "random" else f"random s{seed}"}
        for cap in CAPS:
            e = [-p["effect"] for p in r if p["off_axis"] <= cap]
            row[f"off≤{cap}"] = max(e) if e else float("nan")
        rows.append(row)
rnd = [r for r in rows if r["method"].startswith("random")]
rows = [r for r in rows if not r["method"].startswith("random")]
rows.append({"method": "*random max(5)*", **{k: max(r[k] for r in rnd) for k in rnd[0] if k != "method"}})
rows.sort(key=lambda x: -x["off≤0.6"] if x["off≤0.6"] == x["off≤0.6"] else 9)
print("-C shift toward rejecting the premise (Jev premise levels, higher = less sycophantic), best admissible dose under each damage cap\n")
print(tabulate(rows, headers="keys", tablefmt="pipe", floatfmt="+.2f"))
