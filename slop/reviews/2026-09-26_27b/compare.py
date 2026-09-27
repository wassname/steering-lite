"""4B vs 27B: same methods, same questions, same judge (PI/Claude, 2026-09-26).

Reads outputs/bsbench/results/{full,27b-full}/points.json, writes compare.md and compare.png next to this file.
Question: does steering still work on the larger model, relative to random directions and prompts?
On-axis ÷ room: on-axis change at the Pareto-best dose divided by how far the bare answers could still move toward that side
(8 − bare level for +C, bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap, not in this number.
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


def cell(r, score="score", ci="ci"):
    if r[score] is None:
        return "—"
    bound = lambda v: "−∞" if v is None else f"{v:+.2f}"
    ci_text = "" if r[ci][0] is None and r[ci][1] is None else f" [{bound(r[ci][0])}, {bound(r[ci][1])}]"
    return f"{r[score]:+.2f}{ci_text}"


def side(r, s):
    b = r["best"][s]
    return "—" if b is None else f"{(b['effect'] if s == '+C' else -b['effect']):+.2f} / {b['off_axis']:.2f}"


lines = ["| method | 4B score [90% CI] | 27B score [90% CI] | 4B on-axis ÷ room [90% CI] | 27B on-axis ÷ room [90% CI] | 27B −C on / off | 27B +C on / off | 27B seeds |",
         "|---|---|---|---|---|---|---|---|"]
for m in sorted(methods, key=lambda m: -(rows["27B"][m]["score"] if rows["27B"][m]["score"] is not None else -9)):
    a, b = rows["4B"][m], rows["27B"][m]
    lines.append(f"| {m} | {cell(a)} | {cell(b)} | {cell(a, 'score_room', 'ci_room')} | {cell(b, 'score_room', 'ci_room')} | {side(b, '-C')} | {side(b, '+C')} | {b['seeds']} |")
text = "\n".join(lines)
(here / "compare.md").write_text(__doc__ + "\n" + text + "\n")
print(text)


def panel(ax, score, ci, lim, xlabel, ylabel, title):
    for m in methods:
        a, b = rows["4B"][m], rows["27B"][m]
        if a[score] is None or b[score] is None:
            continue
        err = lambda r: None if r[ci][0] is None or r[ci][1] is None else [[r[score] - r[ci][0]], [r[ci][1] - r[score]]]
        ax.errorbar(a[score], b[score], xerr=err(a), yerr=err(b), fmt="o", color="#888888" if m == "random" else "#1f5fa8", capsize=3)
        ax.annotate(m, (a[score], b[score]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.plot(lim, lim, color="#bbbbbb", lw=1, zorder=0)
    ax.axhline(0, color="#dddddd", lw=1, zorder=0); ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)


fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 6))
panel(left, "score", "ci", [-0.6, 1.8], "score on Qwen3.5-4B (on − off, weaker side)", "score on Qwen3.5-27B",
      "Score, 4B vs 27B (grey line: equal; bars: 90% CI)")
panel(right, "score_room", "ci_room", [-0.3, 0.9], "on-axis ÷ room on Qwen3.5-4B", "on-axis ÷ room on Qwen3.5-27B",
      "On-axis ÷ room, weaker side (share of the bare answers' room used)")
fig.tight_layout()
fig.savefig(here / "compare.png", dpi=150)
