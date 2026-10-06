"""Functional pipeline test: extract -> calibrate -> steer -> save/load.

Tiny random model, CPU, every registered method. No HF network beyond the
hf-internal-testing tiny LlamaForCausalLM (cached).
"""
from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedConfig
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

import steering_lite as sl
from steering_lite import Vector
from steering_lite.config import REGISTRY, _CONFIG_REGISTRY
from steering_lite.variants.value_gram import (
    SteeredDynamicCache,
    _CacheSteeringLease,
)
from steering_lite.variants.vjp_value import ValueGradientCache, _cache_gradients
from steering_lite.variants.vjp_resid import _activations, _encode, _target_mean
import steering_lite.variants.vjp_resid as vjp_resid_module

TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TINY_QWEN = "wassname/qwen3-5lyr-tiny-random"  # query_steer needs self_attn.q_norm, which Llama lacks
METHODS = [
    "mean_diff", "pca", "topk_clusters", "cosine_gated",
    "sspace", "sspace_pca", "corda_pca", "sspace_ablate", "sspace_scale", "sspace_pool",
    "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering", "random", "value_gram", "vjp_resid", "vjp_value", "query_steer",
    "sink_split", "sink_split_resid", "cache_mean_diff",
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
        "mean_diff":            sl.MeanDiffC(**common),
        "pca":                  sl.PCAC(**common),
        "topk_clusters":        sl.TopKClustersC(**common, k=2),
        "cosine_gated":         sl.CosineGatedC(**common, tau=0.0),
        "sspace":               sl.SSpaceC(**common, r=2),
        "sspace_pca":           sl.SSpacePCAC(**common, r=2),
        "corda_pca":            sl.CordaPCAC(**common, r=2),
        "sspace_ablate":        sl.SSpaceAblateC(**common, r=2),
        "sspace_scale":         sl.SSpaceScaleC(**common, r=2),
        "sspace_pool":          sl.SSpacePoolC(**common, r=2),
        "spherical":            sl.SphericalC(**common),
        "directional_ablation": sl.DirectionalAblationC(**common),
        "chars":                sl.CHaRSC(**common, k=2),
        "linear_act":           sl.LinearAcTC(**common),
        "angular_steering":     sl.AngularSteeringC(**common),
        "random":               sl.RandomC(**common),
        "value_gram":           sl.ValueGramC(**common, r=2),
        "cache_mean_diff":      sl.CacheMeanDiffC(**common),
        "vjp_resid":            sl.VjpResidC(**{**common, "layers": (0,)}, target_layer=1, skip_first=0),
        "vjp_value":            sl.VjpValueC(**{**common, "layers": (0,)}, target_layer=1, skip_first=0),
        "query_steer":          sl.QuerySteerC(**common),
        # nu_scale / r_scale given: skip the in-extraction iso-KL calibration (slow on CPU; exercised by just smoke-bsbench)
        "sink_split":           sl.SinkSplitC(**{**common, "layers": (1, 2)}, nu_scale=2.0),
        "sink_split_resid":     sl.SinkSplitResidC(**{**common, "layers": (1, 2)}, nu_scale=2.0, r_scale=0.5),
    }
    return table[method]


@pytest.fixture(scope="module")
def tiny_model():
    tok = AutoTokenizer.from_pretrained(TINY_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or "<pad>"
    model = AutoModelForCausalLM.from_pretrained(TINY_MODEL, torch_dtype=torch.float32).eval()
    return model, tok


@pytest.fixture(scope="module")
def tiny_qwen():
    tok = AutoTokenizer.from_pretrained(TINY_QWEN)
    return AutoModelForCausalLM.from_pretrained(TINY_QWEN, torch_dtype=torch.float32).eval(), tok


def test_methods_list_covers_registry():
    """A newly registered method must be added to METHODS (and its config below), or this fails."""
    assert set(METHODS) == set(REGISTRY), f"not in METHODS: {set(REGISTRY) - set(METHODS)}; stale: {set(METHODS) - set(REGISTRY)}"


@pytest.mark.parametrize("method", METHODS)
def test_pipeline(method, request, tmp_path):
    """extract + calibrate + steer + save/load. One test per method."""
    model, tok = request.getfixturevalue("tiny_qwen" if method in ("query_steer", "sink_split", "sink_split_resid") else "tiny_model")
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

    def logits():
        if method == "cache_mean_diff":
            return REGISTRY[method].score_continuation(model, prompt, prompt[:, -2:]).float()
        return model(prompt).logits.float()

    with torch.no_grad():
        base_logits = logits()
    with v(model, C=cfg.coeff):
        with torch.no_grad():
            steer_logits = logits()
    diff = (steer_logits - base_logits).abs().max().item()
    assert diff > 1e-6, f"{method}: steering had no effect on logits (diff={diff:.2e})"

    # save/load round-trip: identical logits with same coeff
    path = str(tmp_path / f"{method}.safetensors")
    v.save(path)
    v2 = Vector.load(path)
    with v(model, C=cfg.coeff):
        with torch.no_grad():
            l1 = logits().detach()
    with v2(model, C=cfg.coeff):
        with torch.no_grad():
            l2 = logits().detach()
    err = (l1 - l2).abs().max().item()
    assert err < 1e-4, f"{method}: save/load mismatch err={err:.2e}"


# methods that put per-contrast tensors in `stacked` -> Vector + Vector works
MULTI_OK = ["mean_diff", "sspace", "sspace_pca", "sspace_ablate", "sspace_scale",
            "sspace_pool", "topk_clusters", "random", "value_gram"]
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


def test_value_gram_edits_values_not_keys(tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    cfg = sl.ValueGramC(layers=(1,), r=2, coeff=1.0, dtype=torch.float32)
    vector = sl.train(model, tok, POS, NEG, cfg, batch_size=2, max_length=64)
    prompt = tok("Tell me the truth.", return_tensors="pt")

    with torch.no_grad():
        base = model(**prompt, use_cache=True).past_key_values
    with vector(model, C=1.0):
        with torch.no_grad():
            steered = model(**prompt, use_cache=True).past_key_values

    torch.testing.assert_close(steered.layers[1].keys, base.layers[1].keys, rtol=0, atol=0)
    assert not torch.equal(steered.layers[1].values, base.layers[1].values)
    torch.testing.assert_close(steered.layers[0].values, base.layers[0].values, rtol=0, atol=0)


def test_value_gram_zero_and_signed_symmetry(tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(
        model, tok, POS, NEG,
        sl.ValueGramC(layers=(1,), r=2, dtype=torch.float32),
        batch_size=2, max_length=64,
    )
    prompt = tok("Tell me the truth.", return_tensors="pt")
    with torch.no_grad():
        base = model(**prompt, use_cache=True).past_key_values.layers[1].values
    caches = {}
    for coeff in (0.0, 1.0, -1.0):
        with vector(model, C=coeff):
            with torch.no_grad():
                caches[coeff] = model(**prompt, use_cache=True).past_key_values.layers[1].values
    torch.testing.assert_close(caches[0.0], base, rtol=0, atol=0)
    torch.testing.assert_close(
        caches[1.0] - base, -(caches[-1.0] - base), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("method", ["value_gram", "cache_mean_diff"])
def test_cache_extract_batch_invariant_and_label_swap(method, tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    cfg = {
        "value_gram": sl.ValueGramC(layers=(1,), r=2, dtype=torch.float32),
        "cache_mean_diff": sl.CacheMeanDiffC(layers=(1,), dtype=torch.float32),
    }[method]
    batch1 = sl.train(model, tok, POS, NEG, cfg, batch_size=1, max_length=64)
    batch2 = sl.train(model, tok, POS, NEG, cfg, batch_size=2, max_length=64)
    swapped = sl.train(model, tok, NEG, POS, cfg, batch_size=2, max_length=64)
    torch.testing.assert_close(
        batch1.stacked[1]["c"], batch2.stacked[1]["c"], rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        batch2.stacked[1]["c"], -swapped.stacked[1]["c"], rtol=1e-5, atol=1e-6
    )


def test_value_gram_promotes_existing_prefix_and_detaches(tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(
        model, tok, POS, NEG,
        sl.ValueGramC(layers=(1,), r=2, dtype=torch.float32),
        batch_size=2, max_length=64,
    )
    prefix = tok("Tell me", return_tensors="pt")
    with torch.no_grad():
        ordinary = model(**prefix, use_cache=True).past_key_values
    original_keys = ordinary.layers[1].keys.clone()
    original_values = ordinary.layers[1].values.clone()
    next_token = tok(" the", add_special_tokens=False, return_tensors="pt").input_ids[:, :1]
    with vector(model, C=1.0):
        with torch.no_grad():
            promoted = model(
                input_ids=next_token, past_key_values=ordinary, use_cache=True
            ).past_key_values
    torch.testing.assert_close(
        promoted.layers[1].keys[..., :-1, :], original_keys, rtol=0, atol=0
    )
    assert not torch.equal(promoted.layers[1].values[..., :-1, :], original_values)
    torch.testing.assert_close(ordinary.layers[1].values, original_values, rtol=0, atol=0)

    steered_history = promoted.layers[1].values.clone()
    captured = []
    attention = model.model.layers[1].self_attn
    v_proj = attention.v_proj
    handle = v_proj.register_forward_hook(lambda _m, _a, output: captured.append(output))
    try:
        with torch.no_grad():
            detached = model(
                input_ids=next_token, past_key_values=promoted, use_cache=True
            ).past_key_values
    finally:
        handle.remove()
    raw_value = captured[0].view(
        1, 1, attention.config.num_key_value_heads, attention.head_dim
    ).transpose(1, 2)
    torch.testing.assert_close(detached.layers[1].values[..., :-1, :], steered_history)
    torch.testing.assert_close(detached.layers[1].values[..., -1:, :], raw_value, rtol=0, atol=0)


def test_value_gram_same_vector_can_reattach(tiny_model):
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(
        model, tok, POS, NEG,
        sl.ValueGramC(layers=(1,), r=2, dtype=torch.float32),
        batch_size=2, max_length=64,
    )
    prompt = tok("Tell me", return_tensors="pt")
    next_token = tok(" the", add_special_tokens=False, return_tensors="pt").input_ids[:, :1]
    with vector(model, C=1.0):
        with torch.no_grad():
            cache = model(**prompt, use_cache=True).past_key_values
    history = cache.layers[1].values.clone()
    with vector(model, C=1.0):
        with torch.no_grad():
            continued = model(
                input_ids=next_token, past_key_values=cache, use_cache=True
            ).past_key_values
    torch.testing.assert_close(continued.layers[1].values[..., :-1, :], history, rtol=0, atol=0)


def test_value_gram_formula_and_empty_hybrid_promotion():
    config = PreTrainedConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    )
    ordinary = DynamicCache(config=config)
    direction = torch.tensor([[[3.0, 4.0]]])
    directions = {1: direction}
    lease = _CacheSteeringLease()
    cache = SteeredDynamicCache.promote(
        ordinary, config=config, directions=directions, coeff=0.5, lease=lease
    )
    values = torch.tensor([[[[2.0, -1.0], [-3.0, 4.0]]]])
    unit = direction / direction.norm(dim=-1, keepdim=True)
    projection = torch.einsum("bhtd,khd->bhtk", values, unit)
    scale = projection.abs() * direction.norm(dim=-1).T[None, :, None, :]
    expected = values + 0.5 * torch.einsum("bhtk,khd->bhtd", scale, unit)
    torch.testing.assert_close(cache._edit(values, 1), expected)


@pytest.mark.parametrize("method", ["value_gram", "vjp_value", "cache_mean_diff"])
def test_cache_hybrid_generate(method, tiny_model):
    _, tok = tiny_model
    config = Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=[
            "linear_attention", "linear_attention", "full_attention", "linear_attention",
        ],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    cfg = {
        "vjp_value": sl.VjpValueC(layers=(2,), target_layer=3, skip_first=0, coeff=0.2, dtype=torch.float32),
        "value_gram": sl.ValueGramC(layers=(2,), r=2, coeff=0.2, dtype=torch.float32),
        "cache_mean_diff": sl.CacheMeanDiffC(layers=(2,), coeff=0.2, dtype=torch.float32),
    }[method]
    vector = sl.train(model, tok, POS, NEG, cfg, batch_size=2, max_length=64)
    with vector(model):
        output = model.generate(
            torch.tensor([[1, 4, 5]]), max_new_tokens=2, do_sample=False
        )
    assert output.shape == (1, 5)


def test_vjp_value_gradient_flows_through_real_cache_values(tiny_model):
    """VJP-cache differentiates the target through the actual values returned by
    DynamicCache.update, not through a projection/activation hook.

    The activation hook below never marks any activation as requiring grad (the
    model is detached), so the only graph root is the value-cache input set by
    ValueGradientCache. A nonzero gradient w.r.t. the stored cache tensor is
    therefore proof the path runs through the real full-attention value cache.
    """
    model, tok = tiny_model
    sl.detach(model)
    model.requires_grad_(False)
    prompt = POS[:1]
    layers, target_layer = (0,), 1

    # Discriminator: with only the activation hook (no value-cache graph root)
    # the target cannot require grad, so no hook-only gradient exists.
    encoded = _encode(model, tok, prompt, 64)
    with _activations(model, (target_layer,)) as found:
        model(**encoded)
    assert not found[target_layer].requires_grad, \
        "activation hook must not create a gradient path on its own"

    cotangent = _target_mean(model, tok, prompt, target_layer, 1, 64)
    gradients, _valid = _cache_gradients(
        model, tok, prompt, layers, target_layer, cotangent, 0, 64,
    )
    for layer in layers:
        # The differentiated sources are the real DynamicCache.update storage.
        assert gradients[layer].float().norm().item() > 0, \
            f"layer {layer}: zero gradient through the real value cache"

    # Confirms the returned source is the same tensor DynamicCache stored, and
    # that the graph root is the detached earliest selected incoming value that
    # seeds the frozen-model autograd graph; the stored cache value itself is a
    # valid non-leaf autograd input.
    cache = ValueGradientCache(model.config.get_text_config(), layers)
    with torch.enable_grad(), _activations(model, (target_layer,)) as found:
        model(**encoded, past_key_values=cache, use_cache=True)
        assert cache.sources[0] is cache.layers[0].values
        assert cache.layers[0].values.requires_grad
        grad = torch.autograd.grad(
            found[target_layer], [cache.sources[0]],
            grad_outputs=torch.ones_like(found[target_layer]),
        )[0]
        assert grad.norm().item() > 0


def test_vjp_value_two_source_real_storage_hybrid_exclusion(tiny_model):
    """Two preceding full-attention source layers and a later target.

    A direct real-model observation (3 full-attention layers) showed that
    DynamicCache.update returns concatenated storage, so even the earliest
    selected layer's returned tensor is a non-leaf intermediate; both returned
    cache tensors are still valid `autograd.grad` inputs with finite nonzero
    VJPs. We therefore assert real-storage identity, finite/nonzero gradients
    and expected shapes, and selected/non-selected exclusion - not `is_leaf`.
    The Qwen3.5 hybrid proves non-selected recurrent (linear-attention) layers
    stay excluded while full-attention layers remain the only valid sources.
    """
    _, tok = tiny_model
    config = Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=5,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        layer_types=[
            "full_attention", "full_attention", "linear_attention",
            "full_attention", "linear_attention",
        ],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Qwen3_5ForCausalLM(config).eval()
    sl.detach(model)
    model.requires_grad_(False)
    prompt = POS[:1]
    # Two preceding full-attention source layers (0,1) plus a third (3) all
    # feed the later target; layer 2 is a non-selected recurrent layer sitting
    # between the sources and the target and must stay excluded.
    layers, target_layer = (0, 1, 3), 4

    encoded = _encode(model, tok, prompt, 64)
    expected_shape = (
        1, config.num_key_value_heads, encoded["input_ids"].shape[1], config.head_dim,
    )

    cotangent = _target_mean(model, tok, prompt, target_layer, 1, 64)
    gradients, _valid = _cache_gradients(
        model, tok, prompt, layers, target_layer, cotangent, 0, 64,
    )
    for layer in layers:
        g = gradients[layer].float()
        assert tuple(g.shape) == expected_shape, \
            f"layer {layer}: expected VJP shape {expected_shape}, got {tuple(g.shape)}"
        assert torch.isfinite(g).all(), f"layer {layer}: nonfinite VJP"
        assert g.norm().item() > 0, \
            f"layer {layer}: zero VJP through the real value cache"

    # The returned tensors are the actual DynamicLayer storage DynamicCache
    # keeps, and only full-attention selected layers are differentiated.
    cache = ValueGradientCache(model.config.get_text_config(), layers)
    with torch.enable_grad(), _activations(model, (target_layer,)) as found:
        model(**encoded, past_key_values=cache, use_cache=True)
        assert set(cache.sources) == set(layers)
        for layer in layers:
            assert type(cache.layers[layer]) is DynamicLayer, \
                f"layer {layer}: VJP source must be a full-attention DynamicLayer"
            assert cache.sources[layer] is cache.layers[layer].values, \
                f"layer {layer}: returned tensor is not the DynamicCache storage"
        # recurrent (linear-attention) layers are excluded from the source set
        assert type(cache.layers[2]) is not DynamicLayer
        assert 2 not in cache.sources and 4 not in cache.sources


def test_vjp_class_mean_matches_direct_pinned_estimator_fixture(monkeypatch):
    gradients = iter((
        ({1: torch.tensor([[[1.0, 10.0], [3.0, 30.0], [99.0, 99.0]], [[2.0, 20.0], [4.0, 40.0], [6.0, 60.0]]])}, torch.tensor([[True, True, False], [True, True, True]])),
        ({1: torch.tensor([[[5.0, 50.0], [7.0, 70.0], [99.0, 99.0]]])}, torch.tensor([[True, True, False]])),
    ))
    monkeypatch.setattr(vjp_resid_module, "_batch_gradients", lambda *_args, **_kwargs: next(gradients))

    actual = vjp_resid_module._class_mean_vjp(
        object(), object(), ["a", "b", "c"], (1,), 3, torch.zeros(2), 2, 8, 1,
    )[1]
    per_prompt = torch.stack((
        torch.tensor([2.0, 20.0]),
        torch.tensor([4.0, 40.0]),
        torch.tensor([6.0, 60.0]),
    ))
    torch.testing.assert_close(actual, per_prompt.mean(0))


def test_vjp_resid_matches_pinned_difference_normalization_without_sign_flip(monkeypatch):
    """Numerical fixture for vendored vjp.py: normalize(mean_pos - mean_neg) exactly."""
    class FrozenModel:
        def requires_grad_(self, _enabled):
            return self

    target_means = iter((torch.tensor([2.0, 0.0]), torch.tensor([0.0, 0.0])))
    class_means = iter(({1: torch.tensor([1.0, 2.0])}, {1: torch.tensor([3.0, 1.0])}))
    monkeypatch.setattr(vjp_resid_module, "_blocks", lambda _model: [None] * 4)
    monkeypatch.setattr(vjp_resid_module, "_target_mean", lambda *_args, **_kwargs: next(target_means))
    monkeypatch.setattr(vjp_resid_module, "_class_mean_vjp", lambda *_args, **_kwargs: next(class_means))

    vector = vjp_resid_module.vjp_resid(FrozenModel(), object(), ["positive"], ["negative"], (1,), target_layer=3, skip_first=16)
    expected = torch.tensor([-2.0, 1.0]) / torch.tensor([-2.0, 1.0]).norm()
    torch.testing.assert_close(vector.stacked[1]["v"][0], expected)


def test_vjp_registration_and_config_roundtrip():
    """VJP methods are registered and importable with a working config
    round-trip, and the config classes are exported from the package root.
    """
    for name, cfg_cls in (("vjp_resid", sl.VjpResidC), ("vjp_value", sl.VjpValueC)):
        assert name in REGISTRY, f"{name} missing from runtime REGISTRY"
        assert name in _CONFIG_REGISTRY, f"{name} missing from config registry"
        cfg = cfg_cls(layers=(0,), target_layer=1, skip_first=0, coeff=0.2)
        restored = sl.SteeringConfig.from_dict(cfg.to_dict())
        assert restored.method == name
        assert restored.layers == (0,)
        assert restored.target_layer == 1
        assert restored.skip_first == 0


def test_value_gram_attached_save_load_uses_runtime_buffers(tiny_model, tmp_path):
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(
        model, tok, POS, NEG,
        sl.ValueGramC(layers=(1,), r=2, coeff=1.0, dtype=torch.bfloat16),
        batch_size=2, max_length=64,
    )
    prompt = tok("Tell me the truth.", return_tensors="pt")
    path = str(tmp_path / "kv_cache_attached.safetensors")
    with vector(model):
        with torch.no_grad():
            expected = model(**prompt).logits.float()
        sl.save(model, path)
    sl.load(model, path)
    try:
        with torch.no_grad():
            actual = model(**prompt).logits.float()
    finally:
        sl.detach(model)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_sink_split_float_mask_matches_bool_mask():
    """sink_split's attention must treat an additive float mask (finfo.min for masked keys) like the bool mask, with left padding."""
    from types import SimpleNamespace
    from steering_lite.variants import sink_split as S
    torch.manual_seed(0)
    B, H, KVH, T, d = 2, 4, 2, 7, 8
    q, k, v = torch.randn(B, H, T, d), torch.randn(B, KVH, T, d), torch.randn(B, KVH, T, d)
    pad = torch.tensor([2, 0])  # row 0 is left-padded by 2
    real = torch.arange(T)[None] >= pad[:, None]  # [B, S]
    bool_mask = (torch.ones(T, T, dtype=torch.bool).tril()[None] & real[:, None, :])[:, None]  # [B, 1, T, S]
    float_mask = torch.zeros(bool_mask.shape).masked_fill(~bool_mask, torch.finfo(torch.float32).min)
    module = SimpleNamespace(num_key_value_groups=H // KVH)
    S._ACTIVE[id(module)] = {"u": torch.nn.functional.normalize(torch.randn(KVH, d), dim=-1), "vstar": torch.randn(KVH, d),
                             "eps": 1.0, "C": 2.0, "sink_only": False}
    try:
        attn = S._slot_attention(lambda *a, **k: None)
        out_b, _ = attn(module, q, k, v, bool_mask, scaling=d ** -0.5)
        out_f, _ = attn(module, q, k, v, float_mask, scaling=d ** -0.5)
    finally:
        S._ACTIVE.pop(id(module))
    keep = real[:, :, None, None].expand_as(out_b)  # compare real query rows only (pad rows are unused)
    assert torch.allclose(out_b[keep], out_f[keep], atol=1e-5), (out_b - out_f)[keep].abs().max()


@pytest.mark.parametrize("method", ["mean_diff", "linear_act", "value_gram", "vjp_value", "query_steer", "sink_split"])
def test_user_positions_mask(method, request):
    """only_tokens: empty mask = bare (prefill and cached decode), earlier tokens unchanged, masked span steered. PI/OpenAI."""
    from steering_lite.positions import only_tokens
    model, tok = request.getfixturevalue("tiny_qwen" if method in ("query_steer", "sink_split") else "tiny_model")
    sl.detach(model)
    v = sl.train(model, tok, POS, NEG, _make_cfg(method), batch_size=2, max_length=64)
    ids = tok("Tell me the truth about the moon, please.", return_tensors="pt").input_ids
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, 4:] = True
    with torch.no_grad():
        bare = model(ids).logits
        bare_gen = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0)
        with v(model, C=_make_cfg(method).coeff):
            if method == "sink_split":
                with only_tokens(mask), pytest.raises(AssertionError, match="no per-token form"):
                    model(ids)
                return
            with only_tokens(torch.zeros_like(mask)):
                assert torch.equal(model(ids).logits, bare)
                assert torch.equal(model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=0), bare_gen)
            with only_tokens(mask):
                steered = model(ids).logits
    assert torch.equal(steered[:, :4], bare[:, :4])
    assert (steered[:, 4:] - bare[:, 4:]).abs().max() > 1e-6


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_cache_mean_diff_one_shot_prompt_edit(padding_side, tiny_model):
    """Real-cache formula, padding, and decode lifetime. PI/OpenAI."""
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(model, tok, POS, NEG, sl.CacheMeanDiffC(layers=(1,), dtype=torch.float32), batch_size=2, max_length=64)
    prompts = tok(["Tell me the truth.", "Hello."], padding=True, padding_side=padding_side, return_tensors="pt")
    rows = torch.arange(2)
    last = torch.arange(prompts.input_ids.shape[1]).expand_as(prompts.attention_mask).masked_fill(~prompts.attention_mask.bool(), -1).max(-1).values
    next_ids = torch.tensor([[4], [5]])
    with torch.no_grad():
        bare = model(**prompts, use_cache=True)
        bare_values = bare.past_key_values.layers[1].values.clone()
        bare_next = model(next_ids, past_key_values=bare.past_key_values, use_cache=True)
        for coefficient in (0.0, 2.0, -2.0):
            with vector(model, C=coefficient):
                prefix = model(**prompts, use_cache=True)
                torch.testing.assert_close(prefix.logits, bare.logits, rtol=0, atol=0)
                expected = bare_values.clone()
                expected[rows, :, last, :] += coefficient * vector.stacked[1]["c"].sum(0)
                cache = prefix.past_key_values
                torch.testing.assert_close(cache.layers[1].values, expected, rtol=0, atol=0)
                torch.testing.assert_close(cache.layers[1].keys, bare.past_key_values.layers[1].keys[:, :, :-1], rtol=0, atol=0)
                continued = model(next_ids, past_key_values=cache, use_cache=True)
                torch.testing.assert_close(continued.past_key_values.layers[1].values[:, :, :-1], expected, rtol=0, atol=0)
                torch.testing.assert_close(continued.past_key_values.layers[1].values[:, :, -1:], bare_next.past_key_values.layers[1].values[:, :, -1:], rtol=0, atol=0)
                if coefficient:
                    assert (continued.logits - bare_next.logits).abs().max() > 1e-6
                else:
                    torch.testing.assert_close(continued.logits, bare_next.logits, rtol=0, atol=0)
            after = model(next_ids, past_key_values=cache, use_cache=True)
            torch.testing.assert_close(after.past_key_values.layers[1].values[:, :, :-2], expected, rtol=0, atol=0)


def test_cache_mean_diff_calibration_scores_cached_continuation(tiny_model):
    """One-shot scoring equals streamed decode, and KL detects the edit. PI/OpenAI."""
    model, tok = tiny_model
    sl.detach(model)
    vector = sl.train(model, tok, POS, NEG, sl.CacheMeanDiffC(layers=(1,), coeff=10.0, dtype=torch.float32), batch_size=2, max_length=64)
    prompt = tok("Tell me the truth.", return_tensors="pt").input_ids
    generated = torch.tensor([[4, 5, 6]])
    with torch.no_grad(), vector(model):
        batched = REGISTRY["cache_mean_diff"].score_continuation(model, prompt, generated)
        prefix = model(prompt, use_cache=True)
        cache = prefix.past_key_values
        streamed = [prefix.logits[:, -1:]]
        for token in generated[:, :-1].split(1, dim=1):
            step = model(token, past_key_values=cache, use_cache=True)
            streamed.append(step.logits)
            cache = step.past_key_values
        torch.testing.assert_close(batched, torch.cat(streamed, dim=1), rtol=1e-5, atol=1e-6)
    vector.cfg.coeff = 0.0
    zero = sl.measure_kl(vector, model, tok, [prompt[0]], T=3, device="cpu", show_pbar=False)
    vector.cfg.coeff = 10.0
    steered = sl.measure_kl(vector, model, tok, [prompt[0]], T=3, device="cpu", show_pbar=False)
    assert zero["kl_rms"] < 1e-6
    assert steered["kl_rms"] > zero["kl_rms"] + 1e-7
    assert steered["per_t_p95"][0] < 1e-6
