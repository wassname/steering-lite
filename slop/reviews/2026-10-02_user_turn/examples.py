"""Print user-turn Pareto-best answers next to bare, with Jev ratings and audit. PI/OpenAI."""
import json, sys
from pathlib import Path
d = json.loads(Path("outputs/bsbench/results/user-full/points.json").read_text())
qs = {q["scenario"]: q for q in d["questions"]}
method, side, C, n = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
p = next(p for p in d["points"] if p["method"] == method and p["side"] == side and abs(p["C"] - C) / C < 1e-6)
print(f"## {method} {side} C={p['C']:.4g}: effect {p['effect']:+.3f}, off-axis {p['off_axis']:.3f}, mean damage {p['steered_damage']:.3f}\n")
for q in p["questions"][::max(1, 100 // n)][:n]:
    a = q["audit"]
    audit = "—" if a is None else f"on_target {a['on_target']['probabilities']['yes']:.2f}, fabricates {a['fabricates']['probabilities']['yes']:.2f}"
    print(f"### {q['scenario']} ({q['evidence']}; audit {audit})\n\nQ: {qs[q['scenario']]['prompt']}\n\nFLAW: {qs[q['scenario']]['flaw']}\n\nBARE: {qs[q['scenario']]['bare']}\n\nSTEERED: {q['text']}\n")
