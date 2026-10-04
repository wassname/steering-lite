"""Judge someone else's generations with the judge v4 Jev rubric (no walk needed).

Input jsonl, one row per question and condition:
  {"condition": "vjp -0.7", "scenario": "syco_bullshit_v2_...", "bare": "...", "steered": "..."}
Questions and their known flaws are looked up by scenario in data/bsbench/bullshit_bench_v2.jsonl. Prints one line per condition:
  effect = mean premise_change of steered vs bare, -3..+3, negative = more pushback on the nonsense
  off_axis = mean change from bare in everything else, 0..4; coherent while <= judge.MAX_OFF_AXIS
  bs_lost = BullshitBench score lost, bs(bare) - bs(steered), 0-2 (their rubric, per answer); p_fail and the failure checks are reported

  uv run --extra benchmark python scripts/bsbench/judge_file.py generations.jsonl   # needs OPENROUTER_API_KEY in .env
"""
import json
import sys
from pathlib import Path
from statistics import mean

from data import load_cohort
from judge import FAILURES, MAX_OFF_AXIS, bsb_request, cached, key, p_fail, pair_request, premise_change, refresh

bench = load_cohort()
rows = [json.loads(line) for line in Path(sys.argv[1]).open()]
ask = lambda row, text: bsb_request(bench[row["scenario"]]["prompt"], bench[row["scenario"]]["nonsensical_element"], text)
pair = lambda row: pair_request(bench[row["scenario"]]["prompt"], bench[row["scenario"]]["nonsensical_element"], row["bare"], row["steered"])
requests = {key(r): r for row in rows for r in (ask(row, row["bare"]), ask(row, row["steered"]), pair(row))}
refresh(requests, "file", run=True)
have = cached()

for condition in sorted({row["condition"] for row in rows}):
    at = [(have[key(ask(row, row["bare"]))], have[key(ask(row, row["steered"]))], have[key(pair(row))]) for row in rows if row["condition"] == condition]
    effect = mean(premise_change(p) for _, _, p in at)
    off = mean(p["off_axis"]["score"] for _, _, p in at)
    bs_lost = mean(b["bs_score"]["score"] - s["bs_score"]["score"] for b, s, _ in at)
    checks = " ".join(f"{name}={mean(s[name]['probabilities']['yes'] for _, s, _ in at):.2f}" for name in FAILURES)
    print(f"{condition}\tn={len(at)}\teffect={effect:+.2f}\toff_axis={off:.2f}\tbs_lost={bs_lost:+.2f}\tp_fail={mean(p_fail(s) for _, s, _ in at):.2f}\t{checks}\tcoherent={off <= MAX_OFF_AXIS}")
