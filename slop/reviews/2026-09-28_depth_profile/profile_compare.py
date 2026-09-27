"""Is the depth where the persona contrast forms a fixed % of depth? (PI/Claude, 2026-09-28)
Input: `walk.py --profile` log rows (outputs/logs/profile-{4b,27b,olmo}.log). ratio = |mean(h_pos) - mean(h_neg)| / mean |h| at the
last token, per layer. Output: profile_compare.md (crossing depths) and profile_compare.png (ratio / max vs depth).
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
here = Path(__file__).parent
MODELS = (("Qwen3.5-4B", "profile-4b", "#0072b2"), ("Qwen3.5-27B", "profile-27b", "#d55e00"), ("OLMo-2-32B", "profile-olmo", "#009e73"))
TARGETS = {"Qwen3.5-4B": [(29, "vjp target (default)")], "Qwen3.5-27B": [(61, "vjp target (default)"), (48, "vjp target t48")], "OLMo-2-32B": [(61, "vjp target (default)")]}
lines = ["| model | layers | peak layer (depth) | depth at 50% of peak | depth at 90% of peak | layers from end at 90% | steered band (depth) |", "|---|---|---|---|---|---|---|"]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for name, log, color in MODELS:
    rows = [(int(l), float(r)) for l, r in re.findall(r"^L ?(\d+) d=[0-9.]+ ratio=([0-9.]+)", (ROOT / f"outputs/logs/{log}.log").read_text(), re.M)]
    n = len(rows)
    peak_l, peak = max(rows, key=lambda x: x[1])
    cross = lambda f: next(l for l, r in rows if r >= f * peak)
    lo, hi = int(n * 0.2), int(n * 0.8) - 1
    lines.append(f"| {name} | {n} | L{peak_l} ({peak_l / (n - 1):.2f}) | {cross(0.5) / (n - 1):.2f} (L{cross(0.5)}) | {cross(0.9) / (n - 1):.2f} (L{cross(0.9)}) | {n - 1 - cross(0.9)} | L{lo}-L{hi} ({lo / (n - 1):.2f}-{hi / (n - 1):.2f}) |")
    ax.plot([l / (n - 1) for l, _ in rows], [r / peak for _, r in rows], color=color, lw=2, label=f"{name} ({n} layers, peak ratio {peak:.2f})")
    for t, lab in TARGETS[name]:
        ax.plot(t / (n - 1), dict(rows)[t] / peak, "v" if "t48" in lab else "o", color=color, ms=8, mfc="white" if "t48" in lab else color)
ax.axvspan(0.2, 0.8, color="#eeeeee", zorder=0)
ax.text(0.5, 0.02, "steered band (20-80% of depth)", ha="center", fontsize=8, color="#666666")
ax.set_xlabel("relative depth (layer / last layer)")
ax.set_ylabel("persona contrast / its peak\n(|mean pos - mean neg| / |h|, last token)")
ax.set_ylim(0, 1.1)
handles, labels = ax.get_legend_handles_labels()
handles += [plt.Line2D([], [], marker="o", color="#444444", ls=""), plt.Line2D([], [], marker="v", color="#444444", mfc="white", ls="")]
labels += ["VJP target layer, default (3 from the end)", "VJP target layer 48 (27B run t48)"]
ax.legend(handles, labels, frameon=False, loc="upper left", fontsize=8)
ax.set_title("Where the persona contrast forms. Steering is applied in the grey band;\nthe VJP target is the later layer whose contrast the VJP direction aims to change", fontsize=9)
fig.tight_layout(); fig.savefig(here / "profile_compare.png", dpi=150)
text = "\n".join(lines); print(text)
(here / "profile_compare.md").write_text(__doc__ + "\n" + text + "\n")
