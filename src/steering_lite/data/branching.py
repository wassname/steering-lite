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


# Care vs Traditional/Sanctity (binding) axis. Matches tinymfv airisk
# foundation_coarse labels (Care vs Sanctity/Authority/Loyalty cluster).
# Each pair shares scaffold and differs only on the moral-axis content words,
# so the activation diff isolates the value axis, not phrasing.
PERSONA_PAIRS_TRAD_CARE: list[tuple[str, str]] = [
    ("a deeply traditional person",
     "a deeply caring person"),
    ("someone guided by sacred tradition",
     "someone guided by compassion"),
    ("an AI that prioritizes purity and reverence",
     "an AI that prioritizes preventing harm"),
    ("a person who values sanctity above all",
     "a person who values welfare above all"),
    ("a stern traditionalist",
     "a gentle caregiver"),
    ("a thinker rooted in sacred tradition",
     "a thinker rooted in compassion"),
]

PROMPT_TEMPLATE = "Pretend you're {persona} thinking about a task you've been given."


def _data_path() -> Path:
    # src/steering_lite/data/branching.py -> repo root is parents[3]
    return Path(__file__).resolve().parents[3] / "data" / "branching_suffixes.json"


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

    POS: persona-A system prompt + user_msg + assistant=suffix.
    NEG: persona-B system prompt + same user_msg + same suffix.
    Differ only in persona — the suffix is identical, so the activation diff
    isolates the persona signal.
    """
    rng = random.Random(seed)
    if persona_pairs is None:
        persona_pairs = PERSONA_PAIRS_TRAD_CARE
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
        pos_sys = PROMPT_TEMPLATE.format(persona=rng.choice(pos_personas))
        neg_sys = PROMPT_TEMPLATE.format(persona=rng.choice(neg_personas))
        pos_texts.append(tok.apply_chat_template(
            [{"role": "system", "content": pos_sys},
             {"role": "user", "content": user_msg},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True))
        neg_texts.append(tok.apply_chat_template(
            [{"role": "system", "content": neg_sys},
             {"role": "user", "content": user_msg},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True))
    logger.info(f"Persona-branching pairs: n={len(pos_texts)} from {len(persona_pairs)} persona pairs")
    return pos_texts, neg_texts
