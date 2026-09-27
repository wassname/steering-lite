"""Same methods, same questions, same judge on Qwen3.5-4B, Qwen3.5-27B and OLMo-2-32B-Instruct (PI/Claude, 2026-09-27).

Reads outputs/bsbench/results/{full,27b-full,olmo-full}/points.json, writes compare.md and compare.png next to this file.
score: min over sides of (on − 1 × off) at each side's Pareto-best dose (the headline score).
on-axis ÷ room: on-axis change at the Pareto-best dose ÷ how far the bare answers could still move toward that side
(8 − bare premise level for +C, the bare level for −C), weaker side; damage is handled by the dose choice and the 1.5 cap.
Seeds: 4B 3 per learned method, 27B 3, OLMo 1 (cross-seed cos of the vectors is 0.99+, so extra seeds add little).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
MODELS = (("Qwen3.5-4B", "full"), ("Qwen3.5-27B", "27b-full"), ("OLMo-2-32B", "olmo-full"))
rows = {m: {r["method"]: r for r in json.loads((ROOT / f"outputs/bsbench/results/{d}/points.json").read_text())["summary"]} for m, d in MODELS}
methods = [m for m in ("mean_diff", "chars", "vjp_cache", "vjp_delta", "vjp_delta-t48", "random") if any(m in rows[k] for k, _ in MODELS)]


def cell(r, key, ci):
    if r is None or r[key] is None:
        return "—"
    return f"{r[key]:+.2f}" + ("" if r[ci][0] is None else f" [{r[ci][0]:+.2f}, {r[ci][1]:+.2f}]")


head = "| method | " + " | ".join(f"{m} score | {m} on-axis ÷ room" for m, _ in MODELS) + " |"
lines = [head, "|" + "---|" * (1 + 2 * len(MODELS))]
for method in methods:
    lines.append(f"| {method} | " + " | ".join(f"{cell(rows[m].get(method), 'score', 'ci')} | {cell(rows[m].get(method), 'score_room', 'ci_room')}" for m, _ in MODELS) + " |")
text = "\n".join(lines)
(here / "compare.md").write_text(__doc__ + "\n" + text + "\n")
print(text)

fig, ax = plt.subplots(figsize=(8, 4.2))
marker = {"Qwen3.5-4B": "o", "Qwen3.5-27B": "s", "OLMo-2-32B": "D"}
color = {"Qwen3.5-4B": "#0072b2", "Qwen3.5-27B": "#d55e00", "OLMo-2-32B": "#009e73"}
for j, (m, _) in enumerate(MODELS):
    for i, method in enumerate(methods):
        r = rows[m].get(method)
        if r is None or r["score_room"] is None:
            continue
        y = i + (j - 1) * 0.22
        lo, hi = r["ci_room"]
        if lo is not None:
            ax.plot([lo, hi], [y, y], color=color[m], lw=1.5)
        ax.plot(r["score_room"], y, marker[m], color=color[m], ms=7, label=m if i == 0 or method == methods[0] else None)
ax.axvline(0, color="#bbbbbb", lw=1)
ax.set_yticks(range(len(methods)), [m.replace("vjp_delta-t48", "vjp_delta, target 48\n(27B only)") for m in methods])
ax.invert_yaxis()
ax.set_xlabel("on-axis ÷ room: share of the premise room the bare answers leave, weaker side (higher is better)\nbars: 90% CI")
handles = [plt.Line2D([], [], marker=marker[m], color=color[m], ls="", ms=7) for m, _ in MODELS]
ax.legend(handles, [m for m, _ in MODELS], loc="lower right", frameon=False)
ax.set_title("Same steering methods on three models (BS-bench v2, 100 questions, judge: Jev)", fontsize=10)
fig.tight_layout()
fig.savefig(here / "compare.png", dpi=150)
