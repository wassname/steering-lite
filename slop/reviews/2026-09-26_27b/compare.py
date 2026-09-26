"""4B vs 27B: same methods, same questions, same judge (PI/Claude, 2026-09-26).

Reads outputs/bsbench/results/{full,27b-full}/points.json, writes compare.md and compare.png next to this file.
Question: does steering still work on the larger model, relative to random directions and prompts?
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
sites = {m: json.loads((ROOT / f"outputs/bsbench/results/{d}/points.json").read_text()) for m, d in (("4B", "full"), ("27B", "27b-full"))}
rows = {m: {r["method"]: r for r in s["summary"]} for m, s in sites.items()}
methods = [m for m in rows["27B"] if m in rows["4B"]]


def cell(r):
    if r["score"] is None:
        return "—"
    ci = "" if r["ci"][0] is None else f" [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"
    return f"{r['score']:+.2f}{ci}"


def side(r, s):
    b = r["best"][s]
    return "—" if b is None else f"{(b['effect'] if s == '+C' else -b['effect']):+.2f} / {b['off_axis']:.2f}"


lines = ["| method | 4B score [90% CI] | 27B score [90% CI] | 27B −C on / off | 27B +C on / off | 27B seeds |", "|---|---|---|---|---|---|"]
for m in sorted(methods, key=lambda m: -(rows["27B"][m]["score"] if rows["27B"][m]["score"] is not None else -9)):
    lines.append(f"| {m} | {cell(rows['4B'][m])} | {cell(rows['27B'][m])} | {side(rows['27B'][m], '-C')} | {side(rows['27B'][m], '+C')} | {rows['27B'][m]['seeds']} |")
text = "\n".join(lines)
(here / "compare.md").write_text(__doc__ + "\n" + text + "\n")
print(text)

fig, ax = plt.subplots(figsize=(6.5, 6))
for m in methods:
    a, b = rows["4B"][m], rows["27B"][m]
    if a["score"] is None or b["score"] is None:
        continue
    ax.errorbar(a["score"], b["score"],
                xerr=None if a["ci"][0] is None else [[a["score"] - a["ci"][0]], [a["ci"][1] - a["score"]]],
                yerr=None if b["ci"][0] is None else [[b["score"] - b["ci"][0]], [b["ci"][1] - b["score"]]],
                fmt="o", color="#888888" if m == "random" else "#1f5fa8", capsize=3)
    ax.annotate(m, (a["score"], b["score"]), textcoords="offset points", xytext=(6, 4), fontsize=9)
lim = [-0.6, 1.8]
ax.plot(lim, lim, color="#bbbbbb", lw=1, zorder=0)
ax.axhline(0, color="#dddddd", lw=1, zorder=0); ax.axvline(0, color="#dddddd", lw=1, zorder=0)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("score on Qwen3.5-4B (on − off, weaker side)")
ax.set_ylabel("score on Qwen3.5-27B")
ax.set_title("Same methods, 4B vs 27B (grey line: equal score; bars: 90% CI)", fontsize=10)
fig.tight_layout()
fig.savefig(here / "compare.png", dpi=150)
