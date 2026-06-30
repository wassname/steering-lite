"""Persona-branching contrastive pair construction.

Ported from SSteer (`src/ssteer/extract.py:make_persona_dataset` + `core.py:_load_suffixes`).
We extract steering vectors from POS=persona-A + suffix vs NEG=persona-B + same suffix.
The suffix and user_msg are identical in both branches, so the *only* contrastive
signal is the persona prefix.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from loguru import logger


# Keyword sharpening for moral-story POS situations: the raw foundation label is noisy for the
# non-care foundations (the "authority" label is mostly mundane promise/obligation prudence -- traffic
# jams, overgrown lawns -- with no chain-of-command stakes, so a mean_diff vector off it has an
# ambiguous sign). We further filter POS to situations whose text actually voices the foundation, so
# the contrast direction is unambiguous. This sharpens the STEERING set only; the eval instruments are
# independent, so it is not the circular "select toward the measured foundation" cheat. Only defined
# for foundations that need it; fairness works label-only.
FOUNDATION_KEYWORDS: dict[str, list[str]] = {
    "authority": ["boss", "manager", "supervisor", "officer", "police", "order", "command", "rule",
                  "law", "permission", "obey", "authority", "superior", "captain", "sergeant",
                  "principal", "chief", "official", "duty", "instruction", "regulation", "protocol",
                  "in charge", "allowed", "forbid"],
}


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
# Top template from wassname/persona-steering-template-library (seed-24, score 51.3): binds the
# persona to a behaviour channel, not identity-echo. Used for the short-adjective foundation pairs.
PRIORITIES_TEMPLATE = "Use the priorities of a {persona} person."
PERSONA_LIBRARY_TEMPLATE = "Answer as someone who is {{ persona }}."

# Foundation-target axes for the "better steer": pairs are short, mirrored, enactable dispositions
# (library rules: same domain, no negation-only/abstract-absence poles). Chosen for headroom -- the
# model saturates high on Care/Authority, so we steer toward NON-saturated directions: +purity,
# -authority(individualist), +loyalty, +equality, +/-traditionalism. Candidates; selectivity probe
# (does the target foundation move on MFQ-2/MFV while others stay) decides which to keep.
PERSONA_PAIRS_FOUNDATION: dict[str, tuple[str, str]] = {
    "traditionalist": ("traditionalist", "individualist"),       # binding bundle (authority+purity+loyalty)
    "sanctity":       ("sanctity-minded", "utility-minded"),     # +C raises purity/sanctity
    "authority":      ("authority-respecting", "authority-questioning"),
    "loyalty":        ("group-loyal", "self-interested"),
    "equality":       ("egalitarian", "hierarchical"),
}

PERSONA_REGISTRY: dict[str, tuple[list[tuple[str, str]], str]] = {
    # name -> (persona_pairs, template)
    "authority_care": (PERSONA_PAIRS_AUTHORITY, PROMPT_TEMPLATE),   # original showcase axis
    **{k: ([v], PRIORITIES_TEMPLATE) for k, v in PERSONA_PAIRS_FOUNDATION.items()},
}


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


def make_moralstory_pairs(
    tok,
    *,
    n_pairs: int,
    foundation: str = "fairness",
    thinking: bool = True,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Build (POS, NEG) from wassname/moral_stories_foundations SITUATIONS by foundation label.

    POS = situations whose moral dilemma engages `foundation` (e.g. fairness); NEG = a sample
    BALANCED across the other foundations. The contrastive signal is the situation text itself
    (its moral domain), so we read activations over the prompt and do NOT use the chosen/rejected
    completions -- the direction isolates "this scenario is about fairness" vs "about the other
    foundations", a concrete foundation-domain direction. Balanced NEG (equal draw per other
    foundation) so the diff is fairness-vs-rest, not fairness-vs-care (care dominates the corpus).

    NB the dataset labels the MFT foundations (care/fairness/loyalty/authority/sanctity/liberty);
    fairness is the parent of MFQ-2's equality+proportionality split, so steering
    `foundation='fairness'` is read on MFQ-2 as equality+proportionality movement.
    """
    from datasets import load_dataset

    rng = random.Random(seed)
    ds = load_dataset("wassname/moral_stories_foundations")["train"]
    all_founds = sorted({r["foundation"] for r in ds})
    assert foundation in all_founds, f"foundation={foundation!r} not in {all_founds}"
    pos_rows = [r for r in ds if r["foundation"] == foundation]
    if foundation in FOUNDATION_KEYWORDS:
        pat = re.compile(r"\b(" + "|".join(FOUNDATION_KEYWORDS[foundation]) + r")", re.I)
        kept = [r for r in pos_rows if pat.search(r["prompt"])]
        assert len(kept) >= 32, f"{foundation} keyword filter left only {len(kept)} situations (<32)"
        logger.info(f"Keyword-sharpened {foundation}: {len(kept)}/{len(pos_rows)} situations voice the foundation")
        pos_rows = kept
    n = min(n_pairs, len(pos_rows))
    pos_rows = rng.sample(pos_rows, n)
    # balanced NEG: equal draw from each OTHER foundation, so POS-NEG = foundation vs the rest
    others = [f for f in all_founds if f != foundation]
    per = -(-n // len(others))  # ceil
    neg_rows: list[dict] = []
    for f in others:
        fr = [r for r in ds if r["foundation"] == f]
        neg_rows += rng.sample(fr, min(per, len(fr)))
    neg_rows = rng.sample(neg_rows, n)

    think = "<think>" if thinking else ""
    def templ(r):
        return tok.apply_chat_template(
            [{"role": "user", "content": r["prompt"]},
             {"role": "assistant", "content": think}],
            tokenize=False, continue_final_message=True)
    pos_texts = [templ(r) for r in pos_rows]
    neg_texts = [templ(r) for r in neg_rows]
    logger.info(f"Moral-story pairs: n={n} {foundation!r} situations vs balanced-other ({len(others)} foundations)")
    return pos_texts, neg_texts


def _jsonl_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _scenario_text(row: dict, path: Path) -> str:
    if "text" in row:
        return row["text"]
    if "prompt" in row:
        return row["prompt"]
    raise KeyError(f"{path}: scenario row has neither 'text' nor 'prompt': {row.keys()}")


def _render_template(template: str, persona: str) -> str:
    if template == "__verbatim_skill_persona__":
        return persona
    if "{{ persona }}" in template:
        return template.replace("{{ persona }}", persona)
    if "{persona}" in template:
        return template.format(persona=persona)
    raise ValueError(f"persona template lacks a persona slot: {template!r}")


def make_persona_library_pairs(
    tok,
    *,
    library_dir: Path,
    n_pairs: int,
    pair_id: str = "dignity_over_authority",
    template: str = PERSONA_LIBRARY_TEMPLATE,
    scenario_path: Path | None = None,
    thinking: bool = True,
    seed: int = 42,
) -> tuple[list[str], list[str], dict]:
    """Build pairs from the validated persona-template-library pool.

    Scenarios are sampled across source files so Machiavelli, AI-risk, Moral Stories,
    Social Chemistry, and the small v2/character sets all get representation.
    """
    rng = random.Random(seed)
    persona_path = library_dir / "data" / "personas" / "persona_pairs_v2_candidates.jsonl"
    matches = [r for r in _jsonl_rows(persona_path) if r["id"] == pair_id]
    if len(matches) != 1:
        raise ValueError(f"{persona_path}: expected exactly one pair_id={pair_id!r}, found {len(matches)}")
    pair = matches[0]
    if template == "__verbatim_skill_persona__":
        pos_persona = pair["pos_persona"]
        neg_persona = pair["neg_persona"]
    else:
        pos_persona = pair["pos"]
        neg_persona = pair["neg"]

    sampled_rows: list[dict] = []
    if scenario_path is not None:
        rows = _jsonl_rows(scenario_path)
        chosen = rows if len(rows) <= n_pairs else rng.sample(rows, n_pairs)
        for row in chosen:
            sampled_rows.append({
                "source_file": row.get("source", scenario_path.stem),
                "id": row["id"],
                "text": _scenario_text(row, scenario_path),
            })
    else:
        scenario_paths = sorted((library_dir / "data" / "scenarios").glob("*.jsonl"))
        if not scenario_paths:
            raise FileNotFoundError(library_dir / "data" / "scenarios")
        per_source = -(-n_pairs // len(scenario_paths))
        for path in scenario_paths:
            rows = _jsonl_rows(path)
            n = min(per_source, len(rows))
            chosen = rng.sample(rows, n)
            for row in chosen:
                sampled_rows.append({
                    "source_file": path.stem,
                    "id": row["id"],
                    "text": _scenario_text(row, path),
                })

    source_counts: dict[str, int] = {}
    if scenario_path is None:
        rng.shuffle(sampled_rows)
        sampled_rows = sampled_rows[:n_pairs]
    for row in sampled_rows:
        source_counts[row["source_file"]] = source_counts.get(row["source_file"], 0) + 1

    think = "<think>" if thinking else ""
    pos_texts: list[str] = []
    neg_texts: list[str] = []
    for row in sampled_rows:
        pos_user = _render_template(template, pos_persona) + "\n\n" + row["text"]
        neg_user = _render_template(template, neg_persona) + "\n\n" + row["text"]
        pos_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": pos_user},
             {"role": "assistant", "content": think}],
            tokenize=False, continue_final_message=True))
        neg_texts.append(tok.apply_chat_template(
            [{"role": "user", "content": neg_user},
             {"role": "assistant", "content": think}],
            tokenize=False, continue_final_message=True))

    meta = {
        "library_dir": str(library_dir),
        "scenario_path": str(scenario_path) if scenario_path is not None else None,
        "pair_id": pair_id,
        "pos_persona": pos_persona,
        "neg_persona": neg_persona,
        "template": template,
        "source_counts": source_counts,
        "sample_ids": [{k: row[k] for k in ("source_file", "id")} for row in sampled_rows[:16]],
    }
    logger.info(f"Persona-library pairs: n={len(pos_texts)} pair={pair_id!r} template={template!r} "
                f"scenario_path={scenario_path} sources={source_counts}")
    return pos_texts, neg_texts, meta


def make_persona_pairs(
    tok,
    *,
    n_pairs: int,
    thinking: bool = True,
    persona_pairs: list[tuple[str, str]] | None = None,
    template: str = PROMPT_TEMPLATE,
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
        pos_user = template.format(persona=rng.choice(pos_personas)) + "\n\n" + user_msg
        neg_user = template.format(persona=rng.choice(neg_personas)) + "\n\n" + user_msg
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
