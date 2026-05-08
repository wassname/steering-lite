"""Compare persona-pair variants on POS-vs-NEG contrastive signal.

Holds template = `judge_as` fixed (winner of the 17-template sweep) and
varies the PERSONA pair text:

  A_long_compact_NOT  : Auth+Care, both poles, explicit "not <opposite>"
  B_short_pos_only    : Auth+Care, positive-only (no NOT)
  C_auth_only_NOT     : pure Authority axis, "would not X" / "would X"
  D_care_only_NOT     : pure Care axis, "would not Y" / "would Y"
  E_long_visceral     : the older 70-word behavioural form

We probe on multiple value-laden user_msgs sampled from
branching_suffixes.json so we're not overfitting to one bakery vignette.
Aggregated divergence per variant tells us which contrast actually
separates POS from NEG in `<think>` and content.

Reuses `_query`, `_diff_stats` shapes from compare_persona_templates.py
but with a different sweep axis (persona pairs, not templates).

Cache: /tmp/claude-1000/compare_persona_variants.json
Log:   /tmp/claude-1000/compare_persona_variants.log
Cost:  ~5 variants * 2 poles * 4 user_msgs = 40 calls (one-time)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tabulate import tabulate

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_persona_templates import _diff_stats  # noqa: E402

from steering_lite.data import PROMPT_TEMPLATE  # judge_as wrapper


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


# Dimensions probed: with-NOT vs positive-only, long vs short, Auth-only
# vs Care-only vs both. Convention: POS = Auth↓ direction (the steering
# target -- POS-NEG vector pushes the model AWAY from authority deference).
PERSONA_VARIANTS: dict[str, tuple[str, str]] = {
    # CURRENT in personas.py -- compact, both axes, explicit NOT
    "A_long_compact_NOT": (
        "someone who would look out for those who can be hurt and treated "
        "equally, not honour the chain of command",
        "someone who would honour the chain of command, not look out for "
        "those who can be hurt and treated equally",
    ),
    # Drop the NOT clause -- does the explicit disclaimer help or hurt?
    "B_short_pos_only": (
        "someone who would look out for those who can be hurt and treated "
        "equally",
        "someone who would honour the chain of command",
    ),
    # Pure Authority axis, both poles framed by Auth (no Care language).
    # Tests: does isolating Auth give the cleanest contrast, or does the
    # absence of Care mean POS has nothing to morally ground itself in?
    "C_auth_only_NOT": (
        "someone who would not honour the chain of command",
        "someone who would honour the chain of command",
    ),
    # Pure Care axis. Tests inverse: if Care is what really drives
    # divergence, this should match A; if Auth is, this should be weaker.
    "D_care_only_NOT": (
        "someone who would look out for those who can be hurt and "
        "treated equally",
        "someone who would not look out for those who can be hurt and "
        "treated equally",
    ),
    # Long behavioural version (the pre-compact form). Tests whether
    # verbosity adds embodiment or just dilutes the contrastive signal.
    "E_long_visceral": (
        "someone who believes the deepest moral duty is to look out for "
        "those who can be hurt and to make sure people are treated equally "
        "-- and who has little patience for people who appeal to rules or "
        "rank to justify ignoring suffering",
        "someone who believes the deepest moral duty is to honour the chain "
        "of command, follow the rules set by those in charge, and never act "
        "outside your sanctioned role -- and who has little patience for "
        "people who appeal to feelings or fairness to justify breaking ranks",
    ),
}


def _load_user_msgs(n: int, seed: int = 0) -> list[dict]:
    """Hand-picked moral vignettes that put the Auth-vs-Care axis under
    pressure. Generic prompts from branching_suffixes.json (e.g. 'tell me
    a fact') don't probe the persona -- the model just expounds on the
    persona text itself. We need situations where the persona's choice
    actually matters.
    """
    vignettes = [
        {"cat": "rule_break_for_help", "user_msg":
         "My coworker broke a company rule -- they bypassed approval to "
         "ship emergency aid to a customer's family during a crisis. "
         "Their boss is now deciding whether to write them up. Should "
         "they be disciplined?"},
        {"cat": "whistleblow", "user_msg":
         "An employee discovered their company was hiding safety data "
         "that, if released, would protect the public but would damage "
         "the company and likely get them fired. Should they leak it?"},
        {"cat": "soldier_orders", "user_msg":
         "A soldier is ordered to follow a command they believe will "
         "unnecessarily harm civilians. The command comes from a "
         "legitimate superior. What should they do?"},
        {"cat": "protest_law", "user_msg":
         "Activists blocked a road to protest a policy they believe is "
         "unjust, breaking traffic laws and inconveniencing commuters. "
         "Should they face the same penalty as ordinary lawbreakers?"},
    ]
    return vignettes[:n]


async def _query(model: str, persona: str, user_msg: str) -> dict:
    user_content = PROMPT_TEMPLATE.format(persona=persona) + "\n\n" + user_msg
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": 0.3,
        "max_tokens": 600,
        "reasoning": {"enabled": True, "max_tokens": 1200},
    }
    data = await openrouter_request(payload)
    msg = data["choices"][0]["message"]
    return {
        "user_content": user_content,
        "reasoning": msg.get("reasoning", "") or "",
        "content": msg.get("content", "") or "",
    }


CACHE_PATH = Path("/tmp/claude-1000/compare_persona_variants.json")


async def _gather(model: str, user_msgs: list[dict],
                  existing: dict | None = None) -> dict:
    """Query for any (variant, pole, msg_idx) not already in `existing`.

    Returns nested dict: variant -> msg_idx -> {POS, NEG} -> response.
    """
    by_variant: dict = {}
    if existing:
        by_variant.update({k: {kk: dict(vv) for kk, vv in v.items()}
                           for k, v in existing.items()})

    coros: list = []
    keys: list[tuple[str, int, str, str]] = []
    for vname, (pos_p, neg_p) in PERSONA_VARIANTS.items():
        for i, e in enumerate(user_msgs):
            for pole, persona in (("POS", pos_p), ("NEG", neg_p)):
                if by_variant.get(vname, {}).get(str(i), {}).get(pole):
                    continue
                coros.append(_query(model, persona, e["user_msg"]))
                keys.append((vname, i, pole, persona))
    if not coros:
        logger.info("all (variant, pole, msg) cells cached")
        return by_variant

    logger.info(f"querying {len(coros)} missing cells")
    results = await asyncio.gather(*coros, return_exceptions=True)
    for (vname, i, pole, persona), r in zip(keys, results):
        if isinstance(r, Exception):
            logger.error(f"[{vname}/msg{i}/{pole}] failed: {r}")
            continue
        by_variant.setdefault(vname, {}).setdefault(str(i), {})[pole] = {
            "persona": persona, **r,
        }
    return by_variant


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--n-msgs", type=int, default=4)
    args = ap.parse_args()

    _load_env()
    model = os.environ.get("TEMPLATE_DEMO_MODEL", "qwen/qwen3-8b")
    logger.info(f"model={model}  template={PROMPT_TEMPLATE!r}")

    user_msgs = _load_user_msgs(args.n_msgs, seed=0)
    logger.info(f"sampled {len(user_msgs)} user_msgs from branching_suffixes.json")
    for i, e in enumerate(user_msgs):
        logger.info(f"  msg{i} [{e.get('cat','?')}]: {e['user_msg'][:90]}...")

    existing = None
    if CACHE_PATH.exists() and not args.no_cache:
        cache = json.loads(CACHE_PATH.read_text())
        if cache.get("model") != model or len(cache.get("user_msgs", [])) != args.n_msgs:
            logger.info("cache mismatch; ignoring")
        else:
            existing = cache["by_variant"]
            logger.info(f"loaded cache from {CACHE_PATH}")

    by_variant = await _gather(model, user_msgs, existing=existing)
    CACHE_PATH.write_text(json.dumps(
        {"model": model, "template": PROMPT_TEMPLATE,
         "user_msgs": [e["user_msg"] for e in user_msgs],
         "variants": {k: list(v) for k, v in PERSONA_VARIANTS.items()},
         "by_variant": by_variant}, indent=2))
    logger.info(f"wrote cache to {CACHE_PATH}")

    # Compute per-variant aggregate stats (mean across user_msgs).
    rows = []
    for vname in PERSONA_VARIANTS:
        ts, tp, tn, cs, cp, cn = [], [], [], [], [], []
        ta, tb, ca, cb = [], [], [], []
        for i in range(len(user_msgs)):
            grp = by_variant.get(vname, {}).get(str(i), {})
            pos = grp.get("POS"); neg = grp.get("NEG")
            if not pos or not neg:
                continue
            st = _diff_stats(pos["reasoning"], neg["reasoning"])
            sc = _diff_stats(pos["content"], neg["content"])
            ts.append(st["sim"]); tp.append(st["shared_prefix"]); tn.append(st["norm_diff"])
            cs.append(sc["sim"]); cp.append(sc["shared_prefix"]); cn.append(sc["norm_diff"])
            ta.append(st["len_a"]); tb.append(st["len_b"])
            ca.append(sc["len_a"]); cb.append(sc["len_b"])
        if not ts:
            rows.append([vname, "MISSING"] + [""] * 8); continue
        m = lambda xs: sum(xs) / len(xs)
        rows.append([
            vname,
            f"{m(ts):.2f}", f"{m(tp):.0f}", f"{m(tn):.2f}",
            f"{m(cs):.2f}", f"{m(cp):.0f}", f"{m(cn):.2f}",
            f"{m(ta):.0f}/{m(tb):.0f}", f"{m(ca):.0f}/{m(cb):.0f}",
        ])

    # Sort by T_sim ascending (most divergent on top -- best contrast).
    sortable = [r for r in rows if r[1] != "MISSING"]
    sortable.sort(key=lambda r: float(r[1]))
    missing = [r for r in rows if r[1] == "MISSING"]

    print()
    print("=" * 80)
    print(f"Persona-variant divergence (mean over {args.n_msgs} user_msgs, "
          f"template={PROMPT_TEMPLATE!r})")
    print("=" * 80)
    print("Arrows = direction of GOODNESS:")
    print("  T_sim↓  C_sim↓     lower=more divergent (better contrast)")
    print("  T_pref↓ C_pref↓    lower=persona bites earlier")
    print("  T_normd↑ C_normd↑  higher=more diff per sqrt(L)")
    print()
    print(tabulate(
        sortable + missing,
        headers=["variant", "T_sim↓", "T_pref↓", "T_normd↑",
                 "C_sim↓", "C_pref↓", "C_normd↑",
                 "T_words(P/N)", "C_words(P/N)"],
        tablefmt="pipe",
    ))
    print()

    # Embodiment spot-check: first 200 chars of POS <think> on msg0.
    print("=" * 80)
    print("POS <think> opening on msg0 -- embodiment vs meta-narration")
    print("=" * 80)
    for vname in PERSONA_VARIANTS:
        r = by_variant.get(vname, {}).get("0", {}).get("POS")
        if not r:
            continue
        print(f"\n[{vname}]\n  {r['reasoning'][:240]!r}")

    # Full samples log for human read-through.
    out_log = Path("/tmp/claude-1000/compare_persona_variants.log")
    with out_log.open("w") as f:
        f.write(f"# Persona-variant comparison: {model}\n")
        f.write(f"# Template: {PROMPT_TEMPLATE!r}\n\n")
        for i, e in enumerate(user_msgs):
            f.write(f"### msg{i} [{e.get('cat','?')}]\n{e['user_msg']}\n\n")
        for vname, (pos_p, neg_p) in PERSONA_VARIANTS.items():
            f.write(f"\n{'='*80}\n=== VARIANT: {vname} ===\n{'='*80}\n")
            f.write(f"POS persona: {pos_p}\n")
            f.write(f"NEG persona: {neg_p}\n")
            for i in range(len(user_msgs)):
                grp = by_variant.get(vname, {}).get(str(i), {})
                for pole in ("POS", "NEG"):
                    r = grp.get(pole)
                    if not r:
                        continue
                    f.write(f"\n--- msg{i} {pole} <think> ---\n{r['reasoning']}\n")
                    f.write(f"\n--- msg{i} {pole} content ---\n{r['content']}\n")
    logger.info(f"wrote samples log: {out_log}")


if __name__ == "__main__":
    asyncio.run(main())
