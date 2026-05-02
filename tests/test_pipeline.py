"""Functional pipeline test: extract -> calibrate -> steer -> save/load.

Tiny random model, CPU, all 11 methods. ~30s total. No HF network beyond the
hf-internal-testing tiny LlamaForCausalLM (cached).
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


def _make_cfg(method: str, layers=(1,)) -> sl.SteeringConfig:
    low = {"spherical", "angular_steering", "linear_act"}
    coeff = 0.1 if method in low else 2.0
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
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32).eval()
    return model, tok


@pytest.mark.parametrize("method", METHODS)
def test_pipeline(method, tiny_model, tmp_path):
    """extract + calibrate + steer + save/load. One test per method."""
    model, tok = tiny_model
    sl.detach(model)

    cfg = _make_cfg(method)
    v = sl.train(model, tok, POS, NEG, cfg, batch_size=2, max_length=64)
    assert isinstance(v, Vector)
    assert any(t.norm().item() > 0 for d in v.state.values() for t in d.values()), \
        f"{method}: all-zero state -- extract broken"

    # calibrate (cheap: 1 prompt, T=5, max_iters=4) — just exercises the path.
    # Pre-tokenize: the tiny-random Llama tokenizer has no chat template.
    calib_ids = [tok(POS[0], return_tensors="pt").input_ids[0]]
    coeff, _hist = sl.calibrate_iso_kl(
        v, model, tok, calib_ids,
        target_kl=1.0, T=5, max_iters=4,
        bracket=(0.05, 4.0), device="cpu",
    )
    assert torch.isfinite(torch.tensor(coeff)), f"{method}: calibrated coeff not finite: {coeff}"

    prompt = tok("Tell me the truth.", return_tensors="pt").input_ids
    with torch.no_grad():
        base_logits = model(prompt).logits.float()
    with v(model, C=cfg.coeff):
        with torch.no_grad():
            steer_logits = model(prompt).logits.float()
    diff = (steer_logits - base_logits).abs().max().item()
    assert diff > 1e-6, f"{method}: steering had no effect on logits (diff={diff:.2e})"

    # save/load round-trip: identical logits with same coeff
    path = str(tmp_path / f"{method}.safetensors")
    v.save(path)
    v2 = Vector.load(path)
    with v(model, C=cfg.coeff):
        with torch.no_grad():
            l1 = model(prompt).logits.detach().float()
    with v2(model, C=cfg.coeff):
        with torch.no_grad():
            l2 = model(prompt).logits.detach().float()
    err = (l1 - l2).abs().max().item()
    assert err < 1e-4, f"{method}: save/load mismatch err={err:.2e}"
