"""Cost per walk, measured on Qwen3.5-4B, and a scaled estimate for larger models.

Measured:
- GPU time per walk: `timing` in walk certificates (walk.py writes load, setup, per-rung and total seconds).
  Only walks run after that change have it (the timed probe: mean_diff seed 3, full cohort).
- Judge cost per walk: sum of `cost_usd` over judge records (OpenRouter's reported cost), per (method, seed),
  for 100-question walks; blind judge the same from blind.jsonl.
Assumed (stated in the output): GPU hourly prices, the time multiplier for larger models, and that judge cost
does not depend on model size (same number of answers; answers are capped at 512 tokens either way).

    python cost.py        # writes outputs/bsbench/results/cost_estimate.md
"""

import collections
import json
from pathlib import Path
from statistics import median

from judge import default_model_dir

ROOT = Path(__file__).resolve().parents[2]
JUDGMENTS = ROOT / "outputs/bsbench/judgments"

# Modal list prices (modal.com/pricing, as quoted by search 2026-09-25): "Nvidia L40S · $0.000542 / sec",
# "Nvidia A100, 80 GB · $0.000694 / sec", "Nvidia H100 SXM5 · $0.001097 / sec"; per GPU-hour below. Excludes CPU/memory
# (~3% of the L40S cost in outputs/logs/modal-billing-2026-09-24.json).
PRICE = {"L40S": 0.000542 * 3600, "A100-80GB": 0.000694 * 3600, "H100": 0.001097 * 3600}
# time per walk relative to 4B on L40S: decode is memory-bound (weights read per token), prefill/KL probe compute-bound
SCALE = [
    # (model, GPU, time multiplier, reason)
    ("Qwen3.5-4B", "L40S", 1.0, "measured"),
    ("Qwen3.5-9B", "L40S", 2.2, "2.2x the weights read per decoded token on the same GPU"),
    ("Qwen3.5-27B", "H100", 2.5, "6.7x weights, H100 has ~3.9x the memory bandwidth and ~2.7x the bf16 FLOPs of L40S"),
]


def walk_costs(model_dir: Path) -> tuple[dict, dict, dict]:
    """Judge + blind $ per (method, seed) over the full cohort, and certificate timings."""
    full = {(c["method"], c["seed"]) for c in map(json.loads, (p.read_text() for p in (model_dir / "walks").glob("*_full.json")))}
    judge, blind = collections.Counter(), collections.Counter()
    for record in map(json.loads, (JUDGMENTS / "judgments.jsonl").open()):
        key = (record["method"], int(record["seed"]))
        if key in full:
            judge[key] += float(record.get("cost_usd") or 0)
    for record in map(json.loads, (JUDGMENTS / "blind.jsonl").open()):
        key = (record["method"], int(record["seed"]))
        if key in full and record.get("rubric") == "blind-concept-v2":
            blind[key] += float(record.get("cost_usd") or 0)
    timing = {}
    for path in (model_dir / "walks").glob("*_full.json"):
        certificate = json.loads(path.read_text())
        if "timing" in certificate:
            timing[certificate["method"], certificate["seed"]] = certificate["timing"] | {"rungs": len(certificate["rungs"])}
    return judge, blind, timing


def main() -> None:
    model_dir = default_model_dir()
    judge, blind, timing = walk_costs(model_dir)
    steered = {k: v for k, v in judge.items() if not k[0].startswith("prompting")}
    judge_per_walk = median(steered.values())
    blind_per_walk = median(v for k, v in blind.items() if k in steered)
    (key, t), = list(timing.items())[:1] or [((None, None), None)]
    assert t is not None, "no walk certificate with timing yet: run a full walk after walk.py records timing"
    gpu_min = t["total_s"] / 60
    summary = json.loads((ROOT / "outputs/bsbench/results/full/points.json").read_text())["summary"]
    top = [row["method"] for row in summary if row["method"] not in ("random", "prompting", "prompting_engineered") and row["score"] is not None][:4]

    lines = [
        "# Cost per walk and larger-model estimate", "",
        "## Measured on Qwen3.5-4B (L40S)", "",
        f"- GPU time, one new 100-question walk with fresh extraction ({key[0]} seed {key[1]}, {t['rungs']} rungs, GPU {t['gpu']}): "
        f"{gpu_min:.1f} min total = load {t['load_s'] / 60:.1f} + setup (bare answers, extraction, C0) {t['setup_s'] / 60:.1f} + rungs {(t['total_s'] - t['load_s'] - t['setup_s']) / 60:.1f}. "
        f"At ${PRICE['L40S']:.2f}/h: ${gpu_min / 60 * PRICE['L40S']:.2f}. Earlier measured billing (~$1.0 per fresh 100-q walk for 10 walks with 12-19 rungs) is consistent once CPU/memory and container start are included.",
        f"- Judge cost per 100-question walk (DeepSeek, OpenRouter cost_usd summed over its cells): median ${judge_per_walk:.2f} over {len(steered)} steered walks "
        f"(range ${min(steered.values()):.2f}-${max(steered.values()):.2f}); blind judge median ${blind_per_walk:.2f}.",
        "", "| walk (full cohort) | judge $ | blind $ |", "|---|---|---|",
        *(f"| {m} s{s} | {judge[m, s]:.2f} | {blind[m, s]:.2f} |" for m, s in sorted(judge)),
        "", "## Estimate for larger models (inference, about 2x uncertain)", "",
        f"Assumptions: GPU prices { {k: round(v, 2) for k, v in PRICE.items()} } per hour (Modal list prices); judge and blind cost unchanged with model size "
        "(same number of answers, 512-token cap); time multipliers below.", "",
        f"Plan = top 4 methods from full/index.md ({', '.join(top)}) x 3 seeds + 10 random + 2 prompts = 24 walks on 100 questions.", "",
        "| model | GPU | time x | GPU min/walk | GPU $/walk | judge+blind $/walk | total $/walk | plan (24 walks) | reason |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for model, gpu, mult, reason in SCALE:
        minutes = gpu_min * mult
        gpu_cost = minutes / 60 * PRICE[gpu]
        per_walk = gpu_cost + judge_per_walk + blind_per_walk
        lines.append(f"| {model} | {gpu} | {mult:g} | {minutes:.0f} | {gpu_cost:.2f} | {judge_per_walk + blind_per_walk:.2f} | {per_walk:.2f} | {24 * per_walk:.0f} | {reason} |")
    out = ROOT / "outputs/bsbench/results/cost_estimate.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
