"""Pilot rows for the judge v4 pair rubric: 30 9B questions per condition, A/B-swapped copies, and bare vs itself. PI/OpenAI."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/bsbench")
from data import read_answers

D = Path("outputs/bsbench/Qwen--Qwen3.5-9B-g2351502a/answers"); bare = read_answers(D / "bare/bare.jsonl")
S = sorted(bare)[:30]
conds = {"prompt -C": "prompting_s0/-C_C1", "prompt +C": "prompting_s0/+C_C1", "vjp -C 0.5": "vjp_resid_s0/-C_C0.5", "vjp +C 0.315": "vjp_resid_s0/+C_C0.3149802625",
         "vjp -C 1.0": "vjp_resid_s0/-C_C1", "mean_diff -C 2": "mean_diff_s0/-C_C2"}
out = []
for name, f in conds.items():
    a = read_answers(D / f"{f}.jsonl")
    for s in S:
        out.append({"condition": name, "scenario": s, "bare": bare[s]["text"], "steered": a[s]["text"]})
        if name in ("vjp -C 0.5", "vjp +C 0.315", "prompt -C"):
            out.append({"condition": name + " SWAPPED", "scenario": s, "bare": a[s]["text"], "steered": bare[s]["text"]})
out += [{"condition": "bare vs bare", "scenario": s, "bare": bare[s]["text"], "steered": bare[s]["text"]} for s in S]
Path("slop/reviews/2026-10-05_judge_v4/pilot.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
print(len(out))
