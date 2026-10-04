"""Is a persona pair's -C gain BullshitBench detection, or a refusal direction / persona echo? PI/OpenAI 2026-10-04.

For each axis report dir (results.py output) at its scored -C dose (index best -C):
- refusal: mean Jev P(refuses_or_moralizes) steered vs bare, and the BS gain on answers with P(refuses) < 0.5
- echo: share of answers containing the -C persona's distinctive phrases, steered vs bare, and the BS gain on
  answers that contain none of them

  cd scripts/bsbench && uv run --extra benchmark python ../../slop/reviews/2026-10-04_eval_v3/echo_refusal_check.py heldout rubric:DIR credulous:DIR ...
"""
import json
import re
import sys
from pathlib import Path
from statistics import mean

from data import COHORTS, load_cohort, read_answers
from judge import bsb_request, cached, key

ECHO = {  # distinctive words of each -C persona sentence (data.AXES)
    "rubric": r"as someone who|explains? why|does(?:n't| not) make sense",
    "credulous": r"skeptic",
    "sycophantic": r"skeptic",
}
cohort = sys.argv[1]
have = cached()
questions = load_cohort(cohort)
print("| axis | −C dose | BS gain | P(refuses) steered / bare | BS gain where P(refuses) < 0.5 (n) | echo rate steered / bare | BS gain without echo (n) | 'premise' rate steered / bare | BS gain without 'premise' (n) |")
print("|---|---|---|---|---|---|---|---|---|")
for arg in sys.argv[2:]:
    axis, report = arg.split(":")
    site = json.loads((Path(report) / "points.json").read_text())
    row = next(r for r in site["summary"] if r["method"] == "mean_diff")
    best = row["best"]["-C"]
    point = next(p for p in site["points"] if p["method"] == "mean_diff" and p["side"] == "-C" and p["C"] == best["C"])
    model_dir = Path(report).parents[1] / site["model_dir"]
    bare = read_answers(model_dir / "answers/bare/bare.jsonl")
    rows = []
    for q in point["questions"]:
        prompt, flaw = questions[q["scenario"]]["prompt"], questions[q["scenario"]]["nonsensical_element"]
        b = have[key(bsb_request(prompt, flaw, bare[q["scenario"]]["text"]))]
        s = have[key(bsb_request(prompt, flaw, q["text"]))]
        rows.append({"gain": s["bs_score"]["score"] - b["bs_score"]["score"],
                     "ref_s": s["refuses_or_moralizes"]["probabilities"]["yes"], "ref_b": b["refuses_or_moralizes"]["probabilities"]["yes"],
                     "echo_s": bool(re.search(ECHO[axis], q["text"], re.I)), "echo_b": bool(re.search(ECHO[axis], bare[q["scenario"]]["text"], re.I)),
                     "prem_s": "premise" in q["text"].lower(), "prem_b": "premise" in bare[q["scenario"]]["text"].lower()})
    no_ref = [r["gain"] for r in rows if r["ref_s"] < 0.5]
    no_echo = [r["gain"] for r in rows if not r["echo_s"]]
    no_prem = [r["gain"] for r in rows if not r["prem_s"]]
    print(f"| {axis} | {best['C']:.3g} | {mean(r['gain'] for r in rows):+.2f} | {mean(r['ref_s'] for r in rows):.2f} / {mean(r['ref_b'] for r in rows):.2f} | "
          f"{mean(no_ref):+.2f} ({len(no_ref)}) | {mean(r['echo_s'] for r in rows):.0%} / {mean(r['echo_b'] for r in rows):.0%} | "
          f"{mean(no_echo) if no_echo else float('nan'):+.2f} ({len(no_echo)}) | {mean(r['prem_s'] for r in rows):.0%} / {mean(r['prem_b'] for r in rows):.0%} | "
          f"{mean(no_prem) if no_prem else float('nan'):+.2f} ({len(no_prem)}) |")
