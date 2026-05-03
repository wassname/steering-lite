"""Weight-SVD S-space steering, cosine-gated (AntiPaSTO arithmetic relaxation).

Standard activation steering adds a constant bias `h <- h + alpha * v`
regardless of input. Weight-SVD S-space steering operates in the SVD basis of
a Linear's *weight matrix*, so the perturbation is input-dependent: tokens
whose S-space representation aligns with the contrastive direction get
pushed; tokens that don't are left alone.

For a Linear `y = x W^T + b` with `W = U S V^T` truncated to top-r, the
whitened S-space coordinates are accessible from EITHER side:

    x V sqrt(S) = (y - b) U / sqrt(S) = x_S          # x_S == y_S

We use the *output* projection at both extract and apply time:

    x_S = (y - b) @ U / sqrt(S)        # [..., r]

Output capture is cheaper than input capture for residual-writer Linears
(e.g. `mlp.down_proj`: d_out=2560 vs d_in=9728 on Qwen3.5-4B). Residual-
reader Linears (q/k/v/up/gate proj) would prefer the input side.

Contrastive direction in S-space (from POS/NEG persona-branching pairs):

    d_S     = mean(x_S^+) - mean(x_S^-)
    d_S_hat = d_S / ||d_S||

Cosine gate (sign-agnostic, matches `cosine_gated.py`):

    gate     = |cos(x_S, d_S_hat)|              in [0, 1]
    delta_S  = alpha * gate * d_S_hat           [..., r]
    delta_y  = (delta_S * sqrt(S)) @ U_r^T      lift S-space delta back to out-space
    y'       = y + delta_y

This is the arithmetic relaxation of AntiPaSTO (Clark, 2026): same S-space
parameterization, but contrastive mean-diff extraction instead of gradient
optimization. Hook target is `mlp.down_proj` per layer (configurable via
`SteeringConfig.target_submodule`); same layer set as other variants.

Why cosine in S-space (not residual d-space): in d=2560 the cosine of two
random vectors concentrates near 0 -- the gate degenerates. In r=8 the
cosine is a meaningful per-token signal; tokens activating dominant S-space
modes get a strong gate, tokens in the tail get near-zero.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class SSpaceC(SteeringConfig):
    method: str = "sspace"
    target_submodule: str = "mlp.down_proj"
    r: int = 8


def _svd_topr(W: Tensor, r: int) -> tuple[Tensor, Tensor]:
    """W [out, in] -> (U_r [out, r], S_r [r]) in float32 on W's device."""
    U, S, _ = torch.linalg.svd(W.float(), full_matrices=False)
    return U[:, :r].contiguous(), S[:r].contiguous()


@register
class SSpace:
    name = "sspace"

    @staticmethod
    def extract(
        pos_outputs: dict[int, Float[Tensor, "n d_out"]],
        neg_outputs: dict[int, Float[Tensor, "n d_out"]],
        cfg: SSpaceC,
        *,
        layer_to_module: dict[int, torch.nn.Module],
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li, y_pos in pos_outputs.items():
            mod = layer_to_module[li]
            W = mod.weight  # [d_out, d_in]
            if cfg.r > min(W.shape):
                raise ValueError(f"r={cfg.r} exceeds rank bound min{tuple(W.shape)}")
            U_r, S_r = _svd_topr(W.detach(), cfg.r)
            sqrtS = S_r.sqrt().contiguous()                  # [r]

            b = mod.bias.detach().float() if mod.bias is not None else None

            y_pos_f = y_pos.float()
            y_neg_f = neg_outputs[li].float()
            if b is not None:
                y_pos_f = y_pos_f - b
                y_neg_f = y_neg_f - b

            xS_pos = (y_pos_f @ U_r) / sqrtS                 # [n, r]
            xS_neg = (y_neg_f @ U_r) / sqrtS
            dS = xS_pos.mean(0) - xS_neg.mean(0)
            dS_hat = dS / (dS.norm() + ε)

            entry = {"U_r": U_r, "sqrtS": sqrtS, "dS_hat": dS_hat}
            if b is not None:
                entry["b"] = b
            out[li] = entry
        return out

    @staticmethod
    def apply(
        block,
        x: Float[Tensor, "b s d_in"],     # unused; kept for hook signature
        y: Float[Tensor, "b s d_out"],
        state: dict[str, Tensor],
        cfg: SSpaceC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = state["U_r"].to(y)
        sqrtS  = state["sqrtS"].to(y)
        dS_hat = state["dS_hat"].to(y)
        y_eff = y - state["b"].to(y) if "b" in state else y

        xS = (y_eff @ U_r) / sqrtS                                  # [b, s, r]
        xS_norm = xS / (xS.norm(dim=-1, keepdim=True) + ε)
        cos = (xS_norm * dS_hat).sum(dim=-1, keepdim=True)          # [b, s, 1]
        gate = cos.abs()                                             # [0, 1]
        delta_S = cfg.coeff * gate * dS_hat                         # [b, s, r]
        delta_y = (delta_S * sqrtS) @ U_r.T                         # [b, s, d_out]
        return y + delta_y
