"""Persona-branching contrastive pair construction."""
from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger


PERSONA_PAIRS_AUTHORITY: list[tuple[str, str]] = [
    (
        "someone who looks after others' wellbeing even when it means defying authority",
        "someone who defers to authority even when others' wellbeing suffers for it",
    ),
]
PROMPT_TEMPLATE = "You think and act like {persona}."


def _data_path() -> Path:
    return Path(__file__).with_name("branching_suffixes_filt.json")


def load_suffixes(thinking: bool = True) -> list[dict]:
    """Load packaged branching-point suffixes."""
    entries = json.loads(_data_path().read_text())
    entries = [entry for entry in entries if entry["suffix"].strip()]
    for entry in entries:
        entry["suffix"] = entry["suffix"].replace("</think>", "").replace("<think>", "").strip()
    entries = [entry for entry in entries if entry["suffix"]]
    assert entries, "no usable packaged branching suffixes"
    if thinking:
        for i, entry in enumerate(entries):
            if i % 2 == 0:
                entry["suffix"] = f"<think>{entry['suffix']}"
    logger.info(f"Loaded {len(entries)} branching suffixes")
    return entries


def make_persona_pairs(
    tok,
    *,
    n_pairs: int,
    thinking: bool = True,
    persona_pairs: list[tuple[str, str]] | None = None,
    template: str = PROMPT_TEMPLATE,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Build matched positive and negative persona prompts from shared suffixes."""
    rng = random.Random(seed)
    persona_pairs = PERSONA_PAIRS_AUTHORITY if persona_pairs is None else persona_pairs
    entries = load_suffixes(thinking=thinking)
    sampled = rng.sample(entries, min(n_pairs, len(entries)))

    pos_texts: list[str] = []
    neg_texts: list[str] = []
    for entry in sampled:
        pos_persona, neg_persona = rng.choice(persona_pairs)
        pos_user = template.format(persona=pos_persona) + "\n\n" + entry["user_msg"]
        neg_user = template.format(persona=neg_persona) + "\n\n" + entry["user_msg"]
        pos_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": pos_user}, {"role": "assistant", "content": entry["suffix"]}],
            tokenize=False,
            continue_final_message=True,
        ))
        neg_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": neg_user}, {"role": "assistant", "content": entry["suffix"]}],
            tokenize=False,
            continue_final_message=True,
        ))
    logger.info(f"Persona-branching pairs: n={len(pos_texts)} from {len(persona_pairs)} persona pairs")
    return pos_texts, neg_texts
