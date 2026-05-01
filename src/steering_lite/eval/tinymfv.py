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


def evaluate_with_vector(model, tok, *, name: str = "scifi", **kwargs):
    """Run tinymfv.evaluate against `model` with whatever steering is currently
    attached. Pass-through wrapper; no behavior of its own.

    Steering must be attached *before* calling this. Idiomatic:
        with v(model, C=...):
            report = evaluate_with_vector(model, tok, name=...)
    """
    from tinymfv import evaluate
    return evaluate(model, tok, name=name, **kwargs)
