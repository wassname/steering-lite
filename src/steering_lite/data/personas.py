"""Persona-branching contrastive pair construction.

Ported from SSteer (`src/ssteer/extract.py:make_persona_dataset` + `core.py:_load_suffixes`).
We extract steering vectors from POS=persona-A + suffix vs NEG=persona-B + same suffix.
The suffix and user_msg are identical in both branches, so the *only* contrastive
signal is the persona prefix.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger


PERSONA_PAIRS_AUTHORITY: list[tuple[str, str]] = [
    # Care+Auth combined axis under value-conflict framing. Each pole names a
    # primary value and accepts the other axis as collateral cost (rather than
    # being explicitly anti the other axis -- avoids extracting a generic
    # "rebel" or "cruel" direction). No negation in either pole.
    #   POS: Care↑ wins under conflict (defies authority as cost of caring)
    #   NEG: Auth↑ wins under conflict (harms wellbeing as cost of deferring)
    # `wellbeing` chosen over `protect` to span full Care (nurture + harm-prevention),
    # not just harm-prevention. `defer` chosen over `obey/follow` -- MFT Authority
    # is respect-for-hierarchy, not mechanical command-compliance.
    # Single pair (n=1): `make_persona_pairs` picks POS/NEG independently with
    # rng.choice, so multi-pair would destroy the contrast structure (sample i
    # could pair pair_a.POS with pair_b.NEG). Cleaner to encode the desired
    # direction once and let suffix-context be the only noise across 256 samples.
    ("someone who looks after others' wellbeing even when it means defying authority",
     "someone who defers to authority even when others' wellbeing suffers for it"),
]

PROMPT_TEMPLATE = "You think and act like {persona}."


def _data_path() -> Path:
    # src/steering_lite/data/personas.py -> repo root is parents[3].
    # Filtered subset (top-200 by POS-vs-NEG persona divergence under
    # the active PROMPT_TEMPLATE+PERSONA_PAIRS_AUTHORITY); regenerate via
    # `scripts/filter_branching_suffixes.py` if the persona/template change.
    return Path(__file__).resolve().parents[3] / "data" / "branching_suffixes_filt.json"


def load_suffixes(thinking: bool = True) -> list[dict]:
    """Load branching-point suffixes. Each entry: {cat, suffix, user_msg}.

    For thinking-mode models (Qwen3, R1), prepend `<think>` to half so the
    extracted direction matches the inference distribution.
    """
    path = _data_path()
    entries = json.loads(path.read_text())
    entries = [e for e in entries if e["suffix"].strip()]
    # Strip raw <think>...</think> blocks: we add <think> ourselves below, and
    # raw blocks break apply_chat_template(continue_final_message=True).
    for e in entries:
        e["suffix"] = e["suffix"].replace("</think>", "").replace("<think>", "").strip()
    entries = [e for e in entries if e["suffix"]]
    assert entries, f"No suffixes found in {path}"
    if thinking:
        for i, e in enumerate(entries):
            if i % 2 == 0:
                e["suffix"] = f"<think>{e['suffix']}"
    logger.info(f"Loaded {len(entries)} branching suffixes from {path.name}")
    return entries


def make_persona_pairs(
    tok,
    *,
    n_pairs: int,
    thinking: bool = True,
    persona_pairs: list[tuple[str, str]] | None = None,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Build (POS, NEG) chat-templated strings.

    POS: persona-A prepended to user_msg + assistant=suffix.
    NEG: persona-B prepended to same user_msg + same suffix.
    Differ only in persona — the suffix is identical, so the activation diff
    isolates the persona signal.

    Persona injected as a user-message prefix (not a `system` role): some
    chat templates raise on system (Gemma) or fold it into user silently
    (Mistral-base, some Llama variants). User-prefix gives identical
    behaviour on every template at the same token count.
    """
    rng = random.Random(seed)
    if persona_pairs is None:
        persona_pairs = PERSONA_PAIRS_AUTHORITY
    pos_personas = [p for p, _ in persona_pairs]
    neg_personas = [n for _, n in persona_pairs]
    entries = load_suffixes(thinking=thinking)
    n = min(n_pairs, len(entries))
    sampled = rng.sample(entries, n)

    pos_texts: list[str] = []
    neg_texts: list[str] = []
    for entry in sampled:
        suffix = entry["suffix"]
        user_msg = entry["user_msg"]
        pos_user = PROMPT_TEMPLATE.format(persona=rng.choice(pos_personas)) + "\n\n" + user_msg
        neg_user = PROMPT_TEMPLATE.format(persona=rng.choice(neg_personas)) + "\n\n" + user_msg
        pos_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": pos_user},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True))
        neg_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": neg_user},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True))
    logger.info(f"Persona-branching pairs: n={len(pos_texts)} from {len(persona_pairs)} persona pairs")
    return pos_texts, neg_texts
