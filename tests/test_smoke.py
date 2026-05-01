"""Smoke test: train -> context-manager use -> save/load round-trip per method.

Exercises the full Vector API with synthetic data (no network, no GPU).
"""
from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite import Vector

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
METHODS = [
    "mean_diff", "mean_centred", "pca", "topk_clusters", "cosine_gated",
    "sspace", "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering",
]

POS = [
    "I always tell the truth.",
    "Honesty is the foundation of trust.",
    "I refuse to deceive others.",
    "I will be transparent about my mistakes.",
]
NEG = [
    "I will lie to get what I want.",
    "Deception is just a tool.",
    "I will hide the truth when convenient.",
    "I prefer flattering lies to harsh truth.",
]


def make_cfg(method: str, layers: tuple[int, ...] = (1,)) -> sl.SteeringConfig:
    low_coeff = {"spherical", "angular_steering", "linear_act"}
    coeff = 0.1 if method in low_coeff else 2.0
    common = dict(layers=layers, coeff=coeff, dtype=torch.float32, seed=0)
    table = {
        "mean_diff":             sl.MeanDiffC(**common),
        "mean_centred":          sl.MeanDiffC(**common, subtract_corpus_mean=True),
        "pca":                   sl.PCAC(**common),
        "topk_clusters":         sl.TopKClustersC(**common, k=2),
        "cosine_gated":          sl.CosineGatedC(**common, tau=0.0),
        "sspace":                sl.SSpaceC(**common, r=2),
        "spherical":             sl.SphericalC(**common),
        "directional_ablation":  sl.DirectionalAblationC(**common),
        "chars":                 sl.CHaRSC(**common, k=2),
        "linear_act":            sl.LinearAcTC(**common),
        "angular_steering":      sl.AngularSteeringC(**common),
    }
    return table[method]


@pytest.fixture(scope="module")
def tiny_model():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<pad>"
    model = AutoModelForCausalLM.from_pretrained(
        TINY_MODEL, torch_dtype=torch.float32
    ).eval()
    return model, tok


@pytest.mark.parametrize("method", METHODS)
def test_vector_pipeline(method: str, tiny_model, tmp_path):
    model, tok = tiny_model
    sl.detach(model)  # guard against leaked state from previous iteration

    cfg = make_cfg(method)
    v = sl.train(model, tok, POS, NEG, cfg, batch_size=2, max_length=64)

    assert isinstance(v, Vector), f"{method}: train() must return Vector"
    assert v.state, f"{method}: empty state after extract"
    assert any(
        t.norm().item() > 0 for d in v.state.values() for t in d.values()
    ), f"{method}: all-zero state -- extract silently broken"

    prompt = tok("Tell me the truth.", return_tensors="pt").input_ids

    with torch.no_grad():
        base_logits = model(prompt).logits.float()

    with v(model, C=cfg.coeff):
        with torch.no_grad():
            steer_logits = model(prompt).logits.float()

    logit_diff = (steer_logits - base_logits).abs().max().item()
    assert logit_diff > 1e-6, f"{method}: steering had no effect on logits (diff={logit_diff:.2e})"

    # save / load round-trip: logits identical
    path = str(tmp_path / f"{method}.safetensors")
    v.save(path)
    v2 = Vector.load(path)

    with v(model, C=cfg.coeff):
        with torch.no_grad():
            logits1 = model(prompt).logits.detach().float()

    with v2(model, C=cfg.coeff):
        with torch.no_grad():
            logits2 = model(prompt).logits.detach().float()

    err = (logits1 - logits2).abs().max().item()
    assert err < 1e-4, f"{method}: save/load mismatch err={err:.2e}"
