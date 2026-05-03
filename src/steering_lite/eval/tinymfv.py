"""Thin adapter over the `tinymfv` package (tiny moral-foundations vignettes).

Lets you run a steering vector through the tinymfv `evaluate` loop without
copying the eval into this repo. Use as:

    from steering_lite.eval.tinymfv import evaluate_with_vector

    v = sl.train(model, tok, pos, neg, sl.MeanDiffC(layers=(15,)))
    with v(model, C=2.0):
        report = evaluate_with_vector(model, tok, name="scifi")
    print(report["table"])

Requires `tinymfv` to be importable (sibling repo `tiny-mcf-vignettes`).
"""
from __future__ import annotations

import torch
from loguru import logger


@torch.no_grad()
def _demo_via_guided(model, tok, user_prompt: str, frame, max_think_tokens: int) -> str:
    """Run tinymfv's guided_rollout (same path eval uses) and return the full
    decoded trace including special tokens, prompt, think, forced </think>, and
    JSON answer. Mirrors what each evaluated cell does."""
    from tinymfv.guided import guided_rollout, choice_token_ids_tf
    res = guided_rollout(
        model, tok,
        user_prompt=user_prompt,
        choice_token_ids=choice_token_ids_tf(tok),
        max_think_tokens=max_think_tokens,
        schema_hint=frame["q"],
        prefill=frame["prefill"],
        verbose=False,
    )
    return res.raw_full_text


def _log_eval_demo_trace(model, tok, name: str, max_think_tokens: int, vector=None) -> None:
    """One real guided_rollout on the first vignette -- same code path as eval.
    If `vector` is given, show paired base + steered traces for the same prompt."""
    from tinymfv.data import load_vignettes
    from tinymfv.core import CONDITIONS, FRAMES
    from ..attach import detach as _detach, attach as _attach

    vignettes = load_vignettes(name)
    if not vignettes:
        logger.warning(f"tinymfv: no vignettes for name={name!r}, skipping demo trace")
        return
    r = vignettes[0]
    cond = next(iter(CONDITIONS))
    frame_name, frame = next(iter(FRAMES.items()))
    user_prompt = r[cond]
    header = (f"vignette={r.get('id','?')} cond={cond} frame={frame_name} "
              f"max_think={max_think_tokens}")

    if vector is None:
        decoded = _demo_via_guided(model, tok, user_prompt, frame, max_think_tokens)
        logger.info(
            "EXPECT: prompt + <think>...</think> + JSON-bool answer; chat template + special tokens visible.\n"
            f"=== EVAL demo trace ({header}) ===\n{decoded}\n=== /EVAL ==="
        )
        return

    _detach(model)
    decoded_base = _demo_via_guided(model, tok, user_prompt, frame, max_think_tokens)
    _attach(model, vector.cfg, vector.state)
    decoded_steer = _demo_via_guided(model, tok, user_prompt, frame, max_think_tokens)
    logger.info(
        f"EXPECT: same prompt under c=0 vs c={vector.cfg.coeff:+.4f}; both "
        "produce coherent <think>+JSON; steered should differ but not collapse.\n"
        f"=== EVAL demo trace ({header}) ===\n"
        f"--- BASE (c=0) ---\n{decoded_base}\n"
        f"--- STEER (c={vector.cfg.coeff:+.4f}) ---\n{decoded_steer}\n"
        f"=== /EVAL ==="
    )


def evaluate_with_vector(model, tok, *, name: str = "scifi", max_think_tokens: int = 64,
                         vector=None, **kwargs):
    """Run tinymfv.evaluate against `model` with whatever steering is currently
    attached. Pass-through wrapper; emits one decoded demo trace before delegating.

    If `vector` is passed, the demo trace shows paired base+steered output for
    one vignette (and `vector` is re-attached before the real eval). Idiomatic:
        report = evaluate_with_vector(model, tok, name=..., vector=v)
    For an unsteered baseline, omit `vector`.
    """
    from tinymfv import evaluate
    _log_eval_demo_trace(model, tok, name=name, max_think_tokens=max_think_tokens, vector=vector)
    return evaluate(model, tok, name=name, max_think_tokens=max_think_tokens, **kwargs)
