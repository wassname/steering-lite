"""Judge someone else's generations with the eval v3 Jev rubric (no walk needed).

Input jsonl, one row per question and condition:
  {"condition": "vjp -0.7", "scenario": "syco_bullshit_v2_...", "bare": "...", "steered": "..."}
Questions and their known flaws are looked up by scenario in data/bsbench/bullshit_bench_v2.jsonl. Prints one line per condition:
  effect = mean BullshitBench score lost, bs(bare) - bs(steered), 0-2 scale, negative = more pushback on the nonsense
  p_fail = mean P(any steering failure) of the steered answers (largest of judge.FAILURES); coherent while <= judge.MAX_FAILURE

  uv run --extra benchmark python scripts/bsbench/judge_file.py generations.jsonl   # needs OPENROUTER_API_KEY in .env
"""
import json
import sys
from pathlib import Path
from statistics import mean

from data import load_cohort
from judge import FAILURES, MAX_FAILURE, bsb_request, cached, key, p_fail, refresh

bench = load_cohort()
rows = [json.loads(line) for line in Path(sys.argv[1]).open()]
requests = {}
for row in rows:
    for text in (row["bare"], row["steered"]):
        request = bsb_request(bench[row["scenario"]]["prompt"], bench[row["scenario"]]["nonsensical_element"], text)
        requests[key(request)] = request
refresh(requests, "file", run=True)
have = cached()

for condition in sorted({row["condition"] for row in rows}):
    rate = lambda row, text: have[key(bsb_request(bench[row["scenario"]]["prompt"], bench[row["scenario"]]["nonsensical_element"], text))]
    at = [(rate(row, row["bare"]), rate(row, row["steered"])) for row in rows if row["condition"] == condition]
    effect = mean(b["bs_score"]["score"] - s["bs_score"]["score"] for b, s in at)
    fail = mean(p_fail(s) for _, s in at)
    checks = " ".join(f"{name}={mean(s[name]['probabilities']['yes'] for _, s in at):.2f}" for name in FAILURES)
    print(f"{condition}\tn={len(at)}\teffect={effect:+.2f}\tp_fail={fail:.2f} (bare {mean(p_fail(b) for b, _ in at):.2f})\t{checks}\tcoherent={fail <= MAX_FAILURE}")
