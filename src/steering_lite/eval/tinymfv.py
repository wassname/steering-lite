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
def _generate_demo(model, tok, prompt: str, max_new_tokens: int):
    device = next(model.parameters()).device
    enc = tok(prompt, return_tensors="pt").to(device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
    return tok.decode(out[0], skip_special_tokens=False)


def _log_eval_demo_trace(model, tok, name: str, max_think_tokens: int, vector=None) -> None:
    """One full guided rollout on the first vignette. If `vector` is given,
    show paired base (no steering) + steered output for the same prompt."""
    from tinymfv.data import load_vignettes
    from tinymfv.core import CONDITIONS, FRAMES
    from ..attach import detach as _detach, attach as _attach

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
    header = (f"vignette={r.get('id','?')} cond={cond} frame={frame} "
              f"max_think={max_think_tokens}")

    if vector is None:
        decoded = _generate_demo(model, tok, prompt, max_think_tokens)
        logger.info(
            "EXPECT: prompt + <think>...</think> + JSON-bool answer; chat template + special tokens visible.\n"
            f"=== EVAL demo trace ({header}) ===\n{decoded}\n=== /EVAL ==="
        )
        return

    # paired: detach -> generate base -> reattach -> generate steered
    _detach(model)
    decoded_base = _generate_demo(model, tok, prompt, max_think_tokens)
    _attach(model, vector.cfg, vector.state)
    decoded_steer = _generate_demo(model, tok, prompt, max_think_tokens)
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
