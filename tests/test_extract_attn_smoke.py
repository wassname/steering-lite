"""Smoke test for token-pooling extraction strategies.

Verifies train_attn(pool=...) works for each pool x method combination on a
tiny llama. Pooling is orthogonal to direction-finding (mean_diff, pca, ...).

Pools: last, mean, attn_v (mean/max/min/hdiff), attn_kq.
attn_* require attn_implementation="eager".
"""
from __future__ import annotations
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_lite import MeanDiffC, PCAC, train_attn

TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


POS = ["good answer A", "good answer B", "good answer C", "good answer D"]
NEG = ["bad answer A", "bad answer B", "bad answer C", "bad answer D"]


@pytest.fixture(scope="module")
def model_tok():
    tok = AutoTokenizer.from_pretrained(TINY)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TINY, torch_dtype=torch.float32, attn_implementation="eager"
    )
    return model, tok


@pytest.mark.parametrize("pool", ["last", "mean", "attn_v", "attn_kq"])
@pytest.mark.parametrize("method_name", ["mean_diff", "pca"])
def test_pool_method_grid(model_tok, pool, method_name):
    model, tok = model_tok
    cfg_cls = {"mean_diff": MeanDiffC, "pca": PCAC}[method_name]
    cfg = cfg_cls(layers=(1,), coeff=1.0)
    vec = train_attn(model, tok, POS, NEG, cfg, pool=pool,
                     batch_size=4, max_length=32)
    v = vec.state[1]["v"]
    assert v.norm() > 1e-6, f"pool={pool} method={method_name}: zero v"


def test_pools_distinct(model_tok):
    """Different pools should produce different directions."""
    model, tok = model_tok
    cfg = MeanDiffC(layers=(1,), coeff=1.0, normalize=False)
    vs = {}
    for pool in ("last", "mean", "attn_v", "attn_kq"):
        vec = train_attn(model, tok, POS, NEG, cfg, pool=pool,
                         batch_size=4, max_length=32)
        vs[pool] = vec.state[1]["v"]
    cos = torch.nn.functional.cosine_similarity(vs["last"], vs["mean"], dim=0)
    assert cos.abs() < 0.999, f"last and mean produced same v (cos={cos:.4f})"


@pytest.mark.parametrize("agg", ["mean", "max", "min", "hdiff"])
def test_attn_v_pair_agg(model_tok, agg):
    model, tok = model_tok
    cfg = MeanDiffC(layers=(1,), coeff=1.0, normalize=False)
    vec = train_attn(model, tok, POS[:2], NEG[:2], cfg,
                     pool="attn_v", pair_agg=agg,
                     batch_size=2, max_length=32)
    assert vec.state[1]["v"].norm() > 1e-8, f"pair_agg={agg}: zero v"
