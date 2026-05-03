"""Regenerate the engineered system prompt for the prompt baseline.

Adapts AxBench's prompt-based steering baseline (Wu et al. 2025, app. J.1):
have an LLM optimise a system prompt for a target concept, treat the
resulting prompt as a strong baseline. AxBench's metric is concept
*presence* in outputs ("does it mention X"); ours is moral-judgment shift
on the Clifford-MFV vignettes. We keep their template structure
(LLM-writes-prompt-for-concept) and swap their objective clause to match
our metric.

Target axis: Clifford Authority. We generate two prompts in one run: a
+C (Auth↓ "indifference to traditional authority") and a -C (Auth↑
"respect for traditional authority"). Each concept slot is single-valenced
with no foils, no comparative trade-offs, no eval hints.

env: OPENROUTER_API_KEY (loaded from any sibling repo's .env).
usage: uv run --extra baseline python scripts/regen_engineered_prompt.py
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request


REPO_ROOT = Path(__file__).resolve().parents[1]

# AxBench https://arxiv.org/pdf/2501.17148. Appendix J.1. Prompt Based steering
# > Our prompt-based model steering baseline is not a zero-shot prompting baseline. Instead of directly prompting LLM to generate steered responses, we first use LLM to generate an enhanced prompt for model steering. Our template is included in the following
# > `Generate a prompt to guide a language model in producing responses. Objective: Direct the model to include content related to [Concept goes here] (the concept) in its responses. Ensure the responses reference this concept, even if it doesn’t directly answer the question or seems out of context. Optionally, provide in-context examples to reinforce this behaviour. Return only the final prompt without any additional text.``


# AxBench J.1 template, with the objective clause adapted from "include
# content related to X" (their concept-presence behaviour) to "adopt this
# worldview as a personal value system" (our value-shift behaviour).
# AxBench's prompt embodies the concept in *output content*; ours embodies it
# in the model's *values/persona*. Critically: the prompt must NOT mention
# evaluation, scoring, or moral-wrongness judgments -- that would be giving
# the model the eval rubric, which we test on. The prompt installs the
# worldview and lets our independent eval probe whether the worldview shifts
# the model's moral judgments.
PROMPT_TEMPLATE = (
    "Generate a system prompt that makes a language model adopt a particular "
    "worldview as its own personal values.\n"
    "Worldview: {concept}.\n"
    "Direct the model to internalise this perspective and reason from it as "
    "if these were its own deeply held values. Do NOT instruct the model to "
    "evaluate, score, or judge anything; do NOT mention moral wrongness, "
    "right vs wrong, or any evaluation task. Just install the worldview. Do "
    "NOT include in-context examples or numbered scenarios -- keep the prompt "
    "short and abstract.\n"
    "Return only the final prompt without any additional text."
)


# Clifford 2025 MFV Authority foundation, framed as two endpoints. Each is
# single-valenced -- no foils, no comparative trade-offs, no eval hints.
# +C (Auth↓): disobedience/disrespect carries no intrinsic moral weight.
# -C (Auth↑): disobedience/disrespect is gravely wrong on its own.
DEFAULT_CONCEPT_POS = (
    "indifference to traditional authority. Disobedience or disrespect toward "
    "bosses, judges, teachers, parents, or institutions does not carry moral "
    "weight on its own"
)
DEFAULT_CONCEPT_NEG = (
    "respect for traditional authority. Disobedience or disrespect toward "
    "bosses, judges, teachers, parents, or institutions carries strong moral "
    "weight on its own"
)


def _load_env() -> None:
    candidates = [
        REPO_ROOT / ".env",
        REPO_ROOT.parent / "tiny-mcf-vignettes" / ".env",
        REPO_ROOT.parent / "daily-dilemmas-self" / ".env",
    ]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            logger.info(f"loaded env from {p}")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. Tried: "
            + ", ".join(str(p) for p in candidates)
        )


async def _generate(concept: str, llm_model: str) -> str:
    meta_prompt = PROMPT_TEMPLATE.format(concept=concept)
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": meta_prompt}],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    logger.info(f"querying {llm_model} for engineered prompt")
    logger.info(f"concept: {concept!r}")
    logger.debug(f"meta-prompt sent:\n{meta_prompt}")
    data = await openrouter_request(payload)
    text = data["choices"][0]["message"]["content"].strip()
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-model", default="openai/gpt-4o-2024-11-20")
    ap.add_argument("--concept-pos", default=DEFAULT_CONCEPT_POS)
    ap.add_argument("--concept-neg", default=DEFAULT_CONCEPT_NEG)
    args = ap.parse_args()

    _load_env()

    async def _both():
        return await asyncio.gather(
            _generate(args.concept_pos, args.llm_model),
            _generate(args.concept_neg, args.llm_model),
        )

    pos_text, neg_text = asyncio.run(_both())

    print()
    print("--- ENGINEERED_PROMPT_POS (Auth↓) ---")
    print(pos_text)
    print("--- /ENGINEERED_PROMPT_POS ---")
    print()
    print("--- ENGINEERED_PROMPT_NEG (Auth↑) ---")
    print(neg_text)
    print("--- /ENGINEERED_PROMPT_NEG ---")
    print()
    print("Paste both into scripts/baseline_engineered_prompt.py.")
    print(f"POS: {len(pos_text)} chars; NEG: {len(neg_text)} chars; model={args.llm_model}")


if __name__ == "__main__":
    main()
