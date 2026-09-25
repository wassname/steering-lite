"""Mean difference projected onto a pooled suppressed-activation subspace.

From wassname/suppressed-activations (suppressed_activation_subspace.py, Qwen3.5-4B layers
early=23, peak=25, output=32 in its causal demo): tokens whose logit-lens score rises from the
early to the peak layer, then falls by the output layer, are "thought but not said".

per prompt, last token:  z_e, z_p, z_o = unembed(rmsnorm(h[early, peak, output]))
                         rise = center(z_p - z_e);  fall = center(z_p - z_o)
                         score = min(relu(rise), relu(fall))                      # [V]
pooled over all persona prompts:  score = mean(score) * frac(score > 0)          # "persistent" rule
S = qr(center(W_U)[topk(score, rank)] * gain)                                    # [d, rank]
v_l = S S^T (mean(h+_l) - mean(h-_l)), unit norm per layer

Deviations from the source repo: the basis is pooled over the persona prompts (one fixed steering
vector), not per sample; the output layer is the last block's output before the final norm.
Hypothesis for sycophancy: the model may represent "this premise is wrong" mid-network and suppress it.
Adapted by PI/OpenAI.
"""

from dataclasses import dataclass

import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks as _blocks
from .vjp_delta import _activations, _encode


@register_config
@dataclass
class SuppressedMeanDiffC(SteeringConfig):
    method: str = "suppressed_mean_diff"
    rank: int = 32
    early_frac: float = 23 / 32
    peak_frac: float = 25 / 32


@torch.no_grad()
def _last_token_states(model, tok, prompts, layers, batch_size, max_length) -> dict[int, Float[Tensor, "n d"]]:
    out = {layer: [] for layer in layers}
    for start in range(0, len(prompts), batch_size):
        encoded = _encode(model, tok, prompts[start : start + batch_size], max_length)
        with _activations(model, layers) as found:
            model(**encoded)
        last = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(last), device=last.device)
        for layer in layers:
            out[layer].append(found[layer][rows, last].float())
    return {layer: torch.cat(values) for layer, values in out.items()}


def _suppressed_basis(model, tok, states: dict[int, Tensor], early: int, peak: int, output: int, rank: int) -> Tensor:
    W_U = model.get_output_embeddings().weight.float()  # [V, d]
    final_norm = model.model.norm
    gain = final_norm(torch.ones(W_U.shape[1], device=W_U.device, dtype=final_norm.weight.dtype)).float()  # rms(1)=1, so this is the gain for any RMSNorm form
    total = torch.zeros(W_U.shape[0], device=W_U.device)
    positive = torch.zeros_like(total)
    n = states[early].shape[0]
    for start in range(0, n, 16):
        h = torch.stack([states[layer][start : start + 16] for layer in (early, peak, output)], dim=1)  # [b 3 d]
        h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6) * gain
        z = h @ W_U.T  # [b 3 V]
        rise = z[:, 1] - z[:, 0]
        fall = z[:, 1] - z[:, 2]
        score = torch.minimum((rise - rise.mean(-1, keepdim=True)).clamp_min(0), (fall - fall.mean(-1, keepdim=True)).clamp_min(0))
        total += score.sum(0)
        positive += (score > 0).float().sum(0)
    persistent = total / n * positive / n
    token_ids = persistent.topk(rank).indices
    logger.info(
        "SHOULD: suppressed tokens look like content the answer withholds (e.g. doubt, correction words), "
        "ELSE the subspace is formatting/noise. layers early={} peak={} output={} top tokens: {}",
        early, peak, output, [tok.decode([i]) for i in token_ids.tolist()],
    )
    directions = (W_U - W_U.mean(0))[token_ids] * gain  # [rank d]
    return torch.linalg.qr(directions.T, mode="reduced").Q  # [d rank]


@register
class SuppressedMeanDiff:
    name = "suppressed_mean_diff"
    extract_from_prompts = True

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        n_blocks = len(_blocks(model))
        early, peak, output = round(cfg.early_frac * n_blocks) - 1, round(cfg.peak_frac * n_blocks) - 1, n_blocks - 1
        needed = tuple(sorted({*cfg.layers, early, peak, output}))
        pos = _last_token_states(model, tok, pos_prompts, needed, batch_size, max_length)
        neg = _last_token_states(model, tok, neg_prompts, needed, batch_size, max_length)
        both = {layer: torch.cat([pos[layer], neg[layer]]) for layer in (early, peak, output)}
        S = _suppressed_basis(model, tok, both, early, peak, output, cfg.rank)
        out = {}
        for layer in cfg.layers:
            v = pos[layer].mean(0) - neg[layer].mean(0)
            kept = S @ (S.T @ v)
            logger.info("layer {} mean_diff energy kept in S: {:.3f} (SHOULD be well above rank/d={:.3f} for a real overlap)",
                        layer, (kept.norm() / v.norm()).square().item(), cfg.rank / v.shape[0])
            out[layer] = {"shared": {}, "stacked": {"v": (kept / kept.norm()).unsqueeze(0)}}
        return out

    @staticmethod
    def apply(_mod, _x, y: Float[Tensor, "b s d"], _shared, stacked, cfg):
        return y + cfg.coeff * stacked["v"].to(y).sum(dim=0)
