"""Filter branching_suffixes.json by POS-vs-NEG persona divergence.

For each user_msg in branching_suffixes.json, run POS and NEG completions
under the active PROMPT_TEMPLATE + PERSONA_PAIRS_AUTHORITY from
personas.py, score word-level divergence, and keep the entries where
the persona has strongest bite.

Reuses `_query` and `_diff_stats` from compare_persona_templates.py.

Output:
- data/branching_suffixes_filt.json -- top-K entries by divergence
- /tmp/claude-1000/filter_branching_scores.tsv -- all scores, sortable
- /tmp/claude-1000/filter_branching_samples.log -- raw POS/NEG samples

Cost: ~550 entries * 2 calls * ~$0.0015 ~ $1.65 on qwen3-8b.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_persona_templates import _diff_stats  # noqa: E402

from openrouter_wrapper.retry import openrouter_request

from steering_lite.data import PERSONA_PAIRS_AUTHORITY, PROMPT_TEMPLATE


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_env() -> None:
    candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "tiny-mcf-vignettes" / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            logger.info(f"loaded env from {p}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY not set")


async def _query(model: str, persona: str, user_msg: str, sem: asyncio.Semaphore,
                 *, reasoning_tokens: int, content_tokens: int) -> dict:
    """Same shape as compare_persona_templates._query but with tunable budgets
    and a concurrency semaphore so we can run hundreds in parallel."""
    user_content = PROMPT_TEMPLATE.format(persona=persona) + "\n\n" + user_msg
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.3,
        "max_tokens": content_tokens,
        "reasoning": {"enabled": True, "max_tokens": reasoning_tokens},
    }
    async with sem:
        data = await openrouter_request(payload)
    msg = data["choices"][0]["message"]
    return {
        "reasoning": msg.get("reasoning", "") or "",
        "content": msg.get("content", "") or "",
    }


async def _score_entry(model: str, entry: dict, sem: asyncio.Semaphore,
                       *, reasoning_tokens: int, content_tokens: int) -> dict:
    pos_persona, neg_persona = PERSONA_PAIRS_AUTHORITY[0]
    user_msg = entry["user_msg"]
    pos, neg = await asyncio.gather(
        _query(model, pos_persona, user_msg, sem,
               reasoning_tokens=reasoning_tokens, content_tokens=content_tokens),
        _query(model, neg_persona, user_msg, sem,
               reasoning_tokens=reasoning_tokens, content_tokens=content_tokens),
    )
    st = _diff_stats(pos["reasoning"], neg["reasoning"])
    sc = _diff_stats(pos["content"], neg["content"])
    return {
        **entry,  # cat, suffix, user_msg
        "pos_reasoning": pos["reasoning"],
        "neg_reasoning": neg["reasoning"],
        "pos_content": pos["content"],
        "neg_content": neg["content"],
        "T_sim": st["sim"], "T_pref": st["shared_prefix"],
        "T_normd": st["norm_diff"],
        "T_words": (st["len_a"], st["len_b"]),
        "C_sim": sc["sim"], "C_pref": sc["shared_prefix"],
        "C_normd": sc["norm_diff"],
        "C_words": (sc["len_a"], sc["len_b"]),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--input", default="data/branching_suffixes.json")
    ap.add_argument("--output", default="data/branching_suffixes_filt.json")
    ap.add_argument("--top-k", type=int, default=200,
                    help="keep top-K entries by lowest T_sim")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--reasoning-tokens", type=int, default=400)
    ap.add_argument("--content-tokens", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None,
                    help="dev: only process first N entries")
    ap.add_argument("--dry-run", action="store_true",
                    help="print plan and exit without API calls")
    args = ap.parse_args()

    _load_env()

    in_path = REPO_ROOT / args.input
    entries = json.loads(in_path.read_text())
    entries = [e for e in entries if e.get("user_msg", "").strip()]
    if args.limit:
        entries = entries[:args.limit]

    logger.info(f"input: {len(entries)} entries from {in_path.name}")
    logger.info(f"persona POS: {PERSONA_PAIRS_AUTHORITY[0][0][:80]}...")
    logger.info(f"persona NEG: {PERSONA_PAIRS_AUTHORITY[0][1][:80]}...")
    logger.info(f"template: {PROMPT_TEMPLATE!r}")
    logger.info(f"calls: {len(entries) * 2} on {args.model}")

    if args.dry_run:
        logger.info("dry-run; exiting before API calls")
        return

    sem = asyncio.Semaphore(args.concurrency)
    coros = [
        _score_entry(args.model, e, sem,
                     reasoning_tokens=args.reasoning_tokens,
                     content_tokens=args.content_tokens)
        for e in entries
    ]
    scored = []
    for i, fut in enumerate(asyncio.as_completed(coros)):
        try:
            r = await fut
            scored.append(r)
            if (i + 1) % 25 == 0:
                logger.info(f"scored {i+1}/{len(entries)}")
        except Exception as e:
            logger.error(f"entry failed: {e}")

    # Sort by T_sim ascending (lowest = most divergent = best)
    scored.sort(key=lambda r: r["T_sim"])

    # Write all-scores TSV
    tsv = Path("/tmp/claude-1000/filter_branching_scores.tsv")
    with tsv.open("w") as f:
        f.write("rank\tcat\tT_sim\tT_pref\tT_normd\tC_sim\tC_pref\tC_normd\t"
                "T_words_pos\tT_words_neg\tuser_msg\n")
        for i, r in enumerate(scored):
            f.write(f"{i}\t{r['cat']}\t{r['T_sim']:.3f}\t{r['T_pref']}\t"
                    f"{r['T_normd']:.2f}\t{r['C_sim']:.3f}\t{r['C_pref']}\t"
                    f"{r['C_normd']:.2f}\t{r['T_words'][0]}\t{r['T_words'][1]}\t"
                    f"{r['user_msg'][:120]!r}\n")
    logger.info(f"wrote scores TSV: {tsv}")

    # Write raw samples log
    samples_log = Path("/tmp/claude-1000/filter_branching_samples.log")
    with samples_log.open("w") as f:
        for i, r in enumerate(scored[:30]):
            f.write(f"\n{'='*80}\n=== rank={i} cat={r['cat']} T_sim={r['T_sim']:.2f} ===\n")
            f.write(f"user_msg: {r['user_msg']}\n\n")
            f.write(f"--- POS think ---\n{r['pos_reasoning'][:500]}\n\n")
            f.write(f"--- NEG think ---\n{r['neg_reasoning'][:500]}\n\n")
            f.write(f"--- POS content ---\n{r['pos_content'][:300]}\n\n")
            f.write(f"--- NEG content ---\n{r['neg_content'][:300]}\n")
    logger.info(f"wrote top-30 samples: {samples_log}")

    # Save filtered subset (keeping only original schema fields, no API outputs)
    keep_keys = {"cat", "suffix", "user_msg"}
    kept = [{k: v for k, v in r.items() if k in keep_keys} for r in scored[:args.top_k]]
    out_path = REPO_ROOT / args.output
    out_path.write_text(json.dumps(kept, indent=2))
    logger.info(f"wrote {len(kept)} entries (top {args.top_k} by T_sim) to {out_path}")

    # Summary by category
    from collections import Counter
    cat_top = Counter(r["cat"] for r in scored[:args.top_k])
    cat_all = Counter(r["cat"] for r in scored)
    print()
    rows = []
    for cat in sorted(cat_all):
        rows.append([cat, cat_all[cat], cat_top[cat],
                     f"{cat_top[cat] / cat_all[cat]:.0%}"])
    print("Category retention in top-K:")
    print(tabulate(rows, headers=["cat", "n_total", "n_top_K", "retain%"],
                   tablefmt="pipe"))

    # T_sim distribution
    print(f"\nT_sim distribution (lower=better):")
    print(f"  min   {scored[0]['T_sim']:.3f}  ({scored[0]['cat']})")
    print(f"  p10   {scored[len(scored)//10]['T_sim']:.3f}")
    print(f"  p50   {scored[len(scored)//2]['T_sim']:.3f}")
    print(f"  p90   {scored[len(scored)*9//10]['T_sim']:.3f}")
    print(f"  max   {scored[-1]['T_sim']:.3f}  ({scored[-1]['cat']})")


if __name__ == "__main__":
    asyncio.run(main())
