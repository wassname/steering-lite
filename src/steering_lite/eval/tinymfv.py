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
def _log_eval_demo_trace(model, tok, name: str, max_think_tokens: int) -> None:
    """One full guided rollout on the first vignette, decoded with special tokens."""
    from tinymfv.data import load_vignettes
    from tinymfv.core import CONDITIONS, FRAMES

    vignettes = load_vignettes(name)
    if not vignettes:
        logger.warning(f"tinymfv: no vignettes for name={name!r}, skipping demo trace")
        return
    r = vignettes[0]
    cond = next(iter(CONDITIONS))
    frame, fr = next(iter(FRAMES.items()))
    user_prompt = f"{r[cond]}\n\n{fr['q']}"
    messages = [{"role": "user", "content": user_prompt}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "<think>\n"
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(
        **enc, max_new_tokens=max_think_tokens, do_sample=False, pad_token_id=pad_id,
    )
    decoded = tok.decode(out[0], skip_special_tokens=False)
    logger.info(
        f"EXPECT: prompt + <think>...think... </think> + answer; chat template + special tokens visible; "
        "answer should be a JSON-bool ({\"choice\": true/false}).\n"
        f"=== EVAL demo trace (vignette={r.get('id','?')} cond={cond} frame={frame} max_think={max_think_tokens}) ===\n"
        f"{decoded}\n=== /EVAL ==="
    )


def evaluate_with_vector(model, tok, *, name: str = "scifi", max_think_tokens: int = 64, **kwargs):
    """Run tinymfv.evaluate against `model` with whatever steering is currently
    attached. Pass-through wrapper; emits one decoded full-rollout demo trace
    before delegating, then runs the real eval.

    Steering must be attached *before* calling this. Idiomatic:
        with v(model, C=...):
            report = evaluate_with_vector(model, tok, name=...)
    """
    from tinymfv import evaluate
    _log_eval_demo_trace(model, tok, name=name, max_think_tokens=max_think_tokens)
    return evaluate(model, tok, name=name, max_think_tokens=max_think_tokens, **kwargs)
