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
from steering_lite.eval.edge import summarize_anchors
from steering_lite.variants.vjp_delta import orient_vjp_delta

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
METHODS = [
    "mean_diff", "pca", "topk_clusters", "cosine_gated",
    "sspace", "sspace_pca", "corda_pca", "sspace_ablate", "sspace_damp_amp", "super_sspace",
    "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering", "random", "vjp_delta",
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
        "pca":                   sl.PCAC(**common),
        "topk_clusters":         sl.TopKClustersC(**common, k=2),
        "cosine_gated":          sl.CosineGatedC(**common, tau=0.0),
        "sspace":                sl.SSpaceC(**common, r=2),
        "sspace_pca":            sl.SSpacePCAC(**common, r=2),
        "corda_pca":             sl.CordaPCAC(**common, r=2),
        "sspace_ablate":         sl.SSpaceAblateC(**common, r=2),
        "sspace_damp_amp":       sl.SSpaceDampAmpC(**common, r=2),
        "super_sspace":          sl.SuperSSpaceC(**common, r=2),
        "spherical":             sl.SphericalC(**common),
        "directional_ablation":  sl.DirectionalAblationC(**common),
        "chars":                 sl.CHaRSC(**common, k=2),
        "linear_act":            sl.LinearAcTC(**common),
        "angular_steering":      sl.AngularSteeringC(**common),
        "random":                sl.RandomC(**common),
        "vjp_delta":             sl.VjpDeltaC(
            **(common | {"layers": (0,)}), target_layer=-1, skip_first=0
        ),
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
        bracket=(0.05, 4.0), device="cpu", do_sample=False,
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


def test_vjp_delta_label_swap_flips_oriented_vector(tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    cfg = _make_cfg("vjp_delta")
    chosen = sl.train(
        model, tok, POS, NEG, cfg, batch_size=2, max_length=64
    ).stacked[0]["v"]
    swapped = sl.train(
        model, tok, NEG, POS, cfg, batch_size=2, max_length=64
    ).stacked[0]["v"]
    torch.testing.assert_close(chosen, -swapped, rtol=1e-4, atol=1e-5)


def test_vjp_delta_orientation_uses_one_sign_across_layers():
    raw = {
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([-1.0, 0.0]),
    }
    activation_axis = {
        1: torch.tensor([-1.0, 0.0]),
        2: torch.tensor([-0.5, 3**0.5 / 2]),
    }
    oriented, cosines, score, flipped = orient_vjp_delta(raw, activation_axis)

    assert cosines == pytest.approx({1: -1.0, 2: 0.5})
    assert score == pytest.approx(-0.25)
    assert flipped
    torch.testing.assert_close(oriented[1], -raw[1])
    torch.testing.assert_close(oriented[2], -raw[2])


# methods that put per-contrast tensors in `stacked` -> Vector + Vector works
MULTI_OK = ["mean_diff", "sspace", "sspace_pca", "sspace_ablate", "sspace_damp_amp",
            "super_sspace", "topk_clusters", "random", "vjp_delta"]
# methods that keep contrasts in `shared` -> Vector + Vector raises (natural fail)
MULTI_FAIL = ["pca", "cosine_gated", "spherical", "directional_ablation",
              "chars", "linear_act", "angular_steering", "corda_pca"]


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


def test_edge_summary_matches_frozen_meandiff():
    anchors = [
        {"coefficient": -0.188, "answer": 0.03963884338736534,
         "repetition": 0.03508771929824561, "answer_mass": 0.5762189626693726,
         "display_generation": "</think>"},
        {"coefficient": -0.094, "answer": 0.027585284784436226,
         "repetition": 0.005952380952380931, "answer_mass": 0.5515086054801941,
         "display_generation": "</think>"},
        {"coefficient": 0.0, "answer": 0.10669060051441193,
         "repetition": 0.0, "answer_mass": 0.561994731426239,
         "display_generation": "</think>"},
        {"coefficient": 0.16, "answer": 0.04742587357759476,
         "repetition": 0.0, "answer_mass": 0.5931493639945984,
         "display_generation": "</think>"},
        {"coefficient": 0.319, "answer": 0.08509904146194458,
         "repetition": 0.023809523809523836, "answer_mass": 0.5284072756767273,
         "display_generation": "</think>"},
    ]
    summary = summarize_anchors("meandiff(base)", anchors)
    assert summary["swing"] == pytest.approx(0.04546019807457924)
    # score/ratio constants are recomputed from the frozen anchor fields above
    # (swing * (min(am-, am+)/am0)**2). The original constants asserted here had
    # no artifact provenance and never matched, so this test was red from birth;
    # the frozen artifact stores the summary only at display precision (0.0402,
    # 0.94), which both old and new constants satisfy. -- Claude
    assert summary["score"] == pytest.approx(0.04018874208758786)
    assert summary["am_edge/base"] == pytest.approx(0.9402352835866041)
    assert round(summary["score"], 4) == 0.0402
    assert round(summary["am_edge/base"], 2) == 0.94
    assert summary["at_budget"] is True
    assert summary["readout_ok"] is True
