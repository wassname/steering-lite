"""Query steering: mean difference of attention queries (wassname/superkv, renamed query-steering).

For each selected attention layer L, capture every head's query after `q_norm` and before RoPE at the last
real token of each prompt, and take the class difference:

$$q^*_L = \\text{mean}(q^+_L) - \\text{mean}(q^-_L) \\in \\mathbb{R}^{H \\times d_{head}}, \\quad \\hat q^*_L = q^*_L / \\|q^*_L\\|_F$$

At runtime add it to every position's query:

$$q \\leftarrow q + \\alpha \\cdot \\hat q^*_L$$

Keys and values are unchanged, so a head can only change which tokens of the current context it reads
(and how sharply); it cannot write new content the way residual steering does.

Differs from the superkv repo, which adds q* at the last token only: here every position is steered
(steering-lite convention; the teacher-forced KL calibration needs every position steered).
Requires `self_attn.q_norm` (Qwen3 / Qwen3.5); on hybrid models select only full-attention layers.

Ref: https://github.com/wassname/superkv (README "Query steering")
"""
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register, register_config
from ..positions import select
from ..target import _get_blocks
from .vjp_resid import _encode


ε = 1e-8


@register_config
@dataclass
class QuerySteerC(SteeringConfig):
    method: str = "query_steer"


def _q_norms(model, layers) -> dict[int, torch.nn.Module]:
    blocks = _get_blocks(model)
    return {layer: blocks[layer].self_attn.q_norm for layer in layers}


@torch.no_grad()
def _last_token_queries(model, tok, prompts, layers, batch_size, max_length) -> dict[int, Float[Tensor, "h d"]]:
    """Mean over prompts of the post-q_norm query at the last real token, per layer: [heads, head_dim]."""
    sums = {layer: 0.0 for layer in layers}
    grabbed = {}
    hooks = [m.register_forward_hook(lambda _m, _i, out, layer=layer: grabbed.__setitem__(layer, out))
             for layer, m in _q_norms(model, layers).items()]
    try:
        for start in range(0, len(prompts), batch_size):
            batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
            model(**batch)
            last = batch["attention_mask"].sum(1) - 1                              # right padding
            rows = torch.arange(last.shape[0], device=last.device)
            for layer in layers:
                sums[layer] = sums[layer] + grabbed[layer][rows, last].float().sum(0)  # [b s h d] -> [h d]
    finally:
        for h in hooks:
            h.remove()
    return {layer: s / len(prompts) for layer, s in sums.items()}


@register
class QuerySteer:
    name = "query_steer"
    extract_from_prompts = True
    cache_intervention = True  # own hooks via install(), no block-output hook

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        layers = tuple(range(len(_get_blocks(model)))) if cfg.layers is None else tuple(cfg.layers)  # None = all layers
        pos = _last_token_queries(model, tok, pos_prompts, layers, batch_size, max_length)
        neg = _last_token_queries(model, tok, neg_prompts, layers, batch_size, max_length)
        out = {}
        for layer in layers:
            q = pos[layer] - neg[layer]
            out[layer] = {"shared": {}, "stacked": {"q": (q / (q.norm() + ε)).unsqueeze(0)}}  # [1, h, d]
        return out

    @staticmethod
    def install(model, cfg, stacked):
        def hook(_m, _i, out, q):
            return select(out + cfg.coeff * q.to(out), out)  # [b s h d] + [h d]
        return [m.register_forward_hook(lambda _m, _i, out, q=stacked[layer]["q"].sum(0): hook(_m, _i, out, q))
                for layer, m in _q_norms(model, stacked).items()]

    @staticmethod
    def apply(_mod, _x, y, _shared, _stacked, _cfg):
        return y
