"""Functional pipeline test: extract -> calibrate -> steer -> save/load.

Tiny random model, CPU, all 11 methods. ~30s total. No HF network beyond the
hf-internal-testing tiny LlamaForCausalLM (cached).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite import Vector

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
METHODS = [
    "mean_diff", "mean_centred", "pca", "topk_clusters", "cosine_gated",
    "sspace", "sspace_ablate", "sspace_damp_amp", "super_sspace",
    "spherical", "directional_ablation", "chars", "linear_act",
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
        "sspace_ablate":         sl.SSpaceAblateC(**common, r=2),
        "sspace_damp_amp":       sl.SSpaceDampAmpC(**common, r=2),
        "super_sspace":          sl.SuperSSpaceC(**common, r=2),
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
    all_tensors = (
        [t for d in v.shared.values()  for t in d.values()] +
        [t for d in v.stacked.values() for t in d.values()]
    )
    assert any(t.norm().item() > 0 for t in all_tensors), \
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


# methods that put per-contrast tensors in `stacked` -> Vector + Vector works
MULTI_OK = ["mean_diff", "sspace", "sspace_ablate", "sspace_damp_amp",
            "super_sspace", "topk_clusters"]
# methods that keep contrasts in `shared` -> Vector + Vector raises (natural fail)
MULTI_FAIL = ["pca", "cosine_gated", "spherical", "directional_ablation",
              "chars", "linear_act", "angular_steering"]


def _train_two(method, model, tok, *, multi: bool = False):
    """Two vectors from disjoint pos/neg halves so the contrasts truly differ.

    For multi-round, sspace-family methods need full-rank basis (r=-1) because
    `topk(r)` mode selection is contrast-dependent and would pick different
    basis subsets per round, violating the shared-basis invariant.
    """
    cfg = _make_cfg(method)
    if multi and hasattr(cfg, "r"):
        cfg = replace(cfg, r=-1)
    v1 = sl.train(model, tok, POS[:2], NEG[:2], cfg, batch_size=2, max_length=64)
    v2 = sl.train(model, tok, POS[2:], NEG[2:], cfg, batch_size=2, max_length=64)
    return cfg, v1, v2


@pytest.mark.parametrize("method", MULTI_OK)
def test_multi_round(method, tiny_model, tmp_path):
    """Vector + Vector cats stacked, applies through hook, save/load round-trips."""
    model, tok = tiny_model
    sl.detach(model)
    cfg, v1, v2 = _train_two(method, model, tok, multi=True)

    v_sum = v1 + v2
    assert v_sum.k_rounds() == 2, f"{method}: expected k=2 after +, got {v_sum.k_rounds()}"

    prompt = tok("Tell me the truth.", return_tensors="pt").input_ids
    with torch.no_grad():
        base = model(prompt).logits.float()
    with v_sum(model, C=cfg.coeff):
        with torch.no_grad():
            steered = model(prompt).logits.float()
    diff = (steered - base).abs().max().item()
    assert diff > 1e-6, f"{method}: k=2 steering had no effect (diff={diff:.2e})"

    path = str(tmp_path / f"{method}_k2.safetensors")
    v_sum.save(path)
    v_loaded = Vector.load(path)
    assert v_loaded.k_rounds() == 2
    with v_loaded(model, C=cfg.coeff):
        with torch.no_grad():
            steered2 = model(prompt).logits.float()
    err = (steered - steered2).abs().max().item()
    assert err < 1e-4, f"{method}: k=2 save/load mismatch err={err:.2e}"


@pytest.mark.parametrize("method", MULTI_FAIL)
def test_multi_round_natural_fail(method, tiny_model):
    """Methods that keep per-contrast tensors in `shared` must raise on +."""
    model, tok = tiny_model
    sl.detach(model)
    _cfg, v1, v2 = _train_two(method, model, tok)
    with pytest.raises(ValueError, match="shared"):
        _ = v1 + v2
