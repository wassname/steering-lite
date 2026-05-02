"""AXBench-style engineered-prompt baseline.

Generates a system prompt via GPT-4o that directs the target model to embody a
moral concept (default: "Sanctity / traditional moral foundation"), then runs
the same tinymfv airisk eval the sweep uses and reports Δlogit per foundation
vs a bare baseline.

Pipeline:
    1. ask GPT-4o (via openrouter_wrapper) to write a system prompt for the
       concept, using the AXBench template (Wu et al. 2025, arxiv 2501.17148).
       Cache result keyed by sha(concept || template) so repeated runs are free.
    2. load `bare.json` from a prior sweep run (or compute bare if missing).
    3. run tinymfv eval with engineered prompt injected as system message.
    4. compute paired Δlogit per foundation + axis_shift; write JSON next to
       the sweep outputs so it can join the leaderboard table.

env: OPENROUTER_API_KEY (loaded from .env files in steering-lite, tinymfv,
or daily-dilemmas-self -- whichever exists). Fail fast if missing.

ref:
    AXBench template: https://arxiv.org/pdf/2501.17148 §3 "Concept-conditioned
    prompts" (Wu et al. 2025).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from loguru import logger
from openrouter_wrapper.retry import openrouter_request
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_lite._quiet import quiet_external_logs
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, FOUNDATION_SHORT,
    axis_shift, baseline_logit_per_foundation, cue,
    dlogit_per_foundation, format_cell,
)
from steering_lite.eval.tinymfv import evaluate_with_vector


REPO_ROOT = Path(__file__).resolve().parents[1]


# AXBench prompt template. Wu et al. 2025, arxiv 2501.17148. Unmodified --
# the point is to use their exact prompt so this baseline is comparable to
# the published method.
AXBENCH_TEMPLATE = (
    "Generate a prompt to guide a language model in producing responses. "
    "Objective: Direct the model to include content related to {concept}. "
    "Avoid being overly broad or vague. The prompt should be a single system "
    "prompt of 2-4 sentences, written in the imperative voice, addressed to "
    "the model. Do not include any explanation or preamble; output ONLY the "
    "system prompt itself."
)

# Default concept matches the steering axis: move the model toward the
# binding/sanctity cluster (vs its baseline care-heavy stance).
DEFAULT_CONCEPT = (
    "the moral foundation of Sanctity (purity, sacredness, traditional "
    "reverence) -- as distinct from Care/Harm reasoning"
)


class _SystemInjectTok:
    """Wraps a tokenizer to inject a system message at chat-template time."""
    def __init__(self, tok, system: str):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_sys", system)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)

    def __call__(self, *args, **kw):
        return self._tok(*args, **kw)

    def apply_chat_template(self, messages, **kw):
        if messages and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self._sys}] + list(messages)
        return self._tok.apply_chat_template(messages, **kw)


def _load_env() -> None:
    """Load OPENROUTER_API_KEY from any sibling repo's .env. Fail fast."""
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


def _cache_key(concept: str, template: str, model: str) -> str:
    h = hashlib.sha256(f"{template}||{concept}||{model}".encode()).hexdigest()
    return h[:16]


async def _generate_prompt(concept: str, llm_model: str, cache: Path) -> str:
    """Generate (or load cached) AXBench engineered system prompt."""
    cache.mkdir(parents=True, exist_ok=True)
    key = _cache_key(concept, AXBENCH_TEMPLATE, llm_model)
    cf = cache / f"{key}.json"
    if cf.exists():
        cached = json.loads(cf.read_text())
        logger.info(f"engineered prompt cached at {cf}")
        return cached["prompt"]

    user_msg = AXBENCH_TEMPLATE.format(concept=concept)
    payload = {
        "model": llm_model,
        "messages": [{"role": "user", "content": user_msg}],
        "temperature": 0.3,
        "max_tokens": 400,
    }
    logger.info(f"querying {llm_model} for engineered prompt on concept: {concept!r}")
    data = await openrouter_request(payload)
    text = data["choices"][0]["message"]["content"].strip()
    # GPT-4o sometimes wraps in quotes; strip if so.
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()
    cf.write_text(json.dumps({
        "concept": concept,
        "llm_model": llm_model,
        "template": AXBENCH_TEMPLATE,
        "prompt": text,
    }, indent=2))
    logger.info(f"engineered prompt saved to {cf} ({len(text)} chars)")
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--llm-model", default="openai/gpt-4o-2024-11-20",
                    help="openrouter model for prompt generation")
    ap.add_argument("--concept", default=DEFAULT_CONCEPT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--vignettes", default="airisk")
    ap.add_argument("--bare-json", type=Path, default=Path("outputs/tinymfv_sweep/bare.json"),
                    help="reuse bare baseline if present (skip re-running base)")
    ap.add_argument("--out", type=Path, default=Path("outputs/tinymfv_sweep"))
    ap.add_argument("--prompt-cache", type=Path, default=Path("outputs/engineered_prompts"))
    args = ap.parse_args()

    quiet_external_logs()
    args.out.mkdir(parents=True, exist_ok=True)

    _load_env()

    # --- 1. generate engineered prompt -------------------------------------
    engineered = asyncio.run(_generate_prompt(args.concept, args.llm_model, args.prompt_cache))
    logger.info(f"engineered system prompt:\n---\n{engineered}\n---")

    # --- 2. load model + tokenizer -----------------------------------------
    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    # --- 3. base report (reuse if present) ---------------------------------
    if args.bare_json.exists():
        logger.info(f"reusing bare report from {args.bare_json}")
        bare = json.loads(args.bare_json.read_text())
        # Reconstruct minimal report shape for foundations.dlogit_per_foundation.
        base_report = {"raw": bare["raw_p_true"]}
    else:
        logger.info("no cached bare report -- running bare eval now")
        t0 = time.time()
        base_report = evaluate_with_vector(
            model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens)
        logger.info(f"bare elapsed={time.time()-t0:.1f}s")

    # --- 4. eval with engineered prompt ------------------------------------
    logger.info("=== engineered_prompt eval ===")
    wrapped_tok = _SystemInjectTok(tok, engineered)
    t0 = time.time()
    eng_report = evaluate_with_vector(
        model, wrapped_tok, name=args.vignettes, max_think_tokens=args.max_think_tokens)
    elapsed = time.time() - t0

    # --- 5. compute Δlogit + axis_shift ------------------------------------
    dlogit = dlogit_per_foundation(base_report, eng_report, args.vignettes)
    axis = axis_shift(dlogit)

    # --- 6. persist + print row --------------------------------------------
    out_path = args.out / "engineered_prompt.json"
    out_path.write_text(json.dumps({
        "label": "engineered_prompt",
        "model": args.model,
        "concept": args.concept,
        "llm_model": args.llm_model,
        "system_prompt": engineered,
        "vignettes": args.vignettes,
        "dlogit_per_foundation": dlogit,
        "axis_shift": axis,
        "raw_p_true": eng_report["raw"],
        "max_think_tokens": args.max_think_tokens,
        "elapsed_s": elapsed,
    }, indent=2))

    from tabulate import tabulate
    headers = (["cue", "axis", "row"]
               + [f"Δ{FOUNDATION_SHORT[f]}" for f in FOUNDATION_ORDER]
               + ["t"])
    row = [cue(axis), f"{axis:+.2f}", "engineered_prompt"]
    row += [format_cell(dlogit[f]) for f in FOUNDATION_ORDER]
    row.append(f"{elapsed:.0f}s")
    logger.info("\n" + tabulate([row], headers=headers, tablefmt="tsv"))
    logger.info(f"out: {out_path}")


if __name__ == "__main__":
    main()
