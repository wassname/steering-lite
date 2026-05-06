"""Multiplicative damp/amp steering in S-space.

Companion to `sspace`: same extract path (full SVD, |dS|.topk(r) mode
selection, store U_r, sqrtS, dS_hat, Vh_r). At apply time, instead of
adding `alpha * gate * d_S_hat` we *multiply* per-mode singular values by
`exp(c * d_S_hat_i)`, so modes with positive contrastive sign get amplified
and modes with negative get damped.

Math (per token):

    For W = U Σ V^T,  y - b = sum_i (x v_i) σ_i u_i.
    y_orig^(mode i)  = (x v_i) σ_i u_i
    y_new^(mode i)   = (x v_i) σ_i exp(c d_S_hat_i) u_i        # multiplicative
    delta_y^(mode i) = (x v_i) σ_i (exp(c d_S_hat_i) - 1) u_i

    Identity: (x v_i) σ_i = (y - b) u_i  (row of (y-b) onto u_i)
    -> delta_y_i = ((y-b) u_i) * (exp(c d_S_hat_i) - 1) * u_i

In matrix form over r selected modes:

    proj_r   = (y - b) @ U_r                                    # [..., r]
    scale    = exp(c * d_S_hat).clamp_(±clamp) - 1              # [r]
    delta_y  = (proj_r * scale) @ U_r^T                         # [..., d_out]
    y'       = y + delta_y

Properties:
  - Monotone in c (larger |c| -> stronger effect).
  - S_eff = σ * exp(...) > 0 always; never sign-flips a mode.
  - At c=0 the steering is exactly identity (scale=0).
  - Non-selected modes contribute 0 (they are absent from U_r).

Compare to:
  - `sspace.py` (additive cosine-gated): adds `alpha * gate * d_S_hat`,
    sign-agnostic gate.
  - `sspace_ablate.py` (subtractive): projects d_S_hat *out* of x_S.

Hook target: `mlp.down_proj` (output-side); V is implicit via SVD identity
so we don't need Vh_r at apply time even though extract saves it.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register
from .sspace import SSpace


ε = 1e-8


@register_config
@dataclass
class SSpaceDampAmpC(SteeringConfig):
    method: str = "sspace_damp_amp"
    r: int = -1  # -1 = full rank; else top-r modes by |d_S|
    clamp_max: float = 4.0  # cap |log_scale| to avoid exp blowup; exp(4) ~ 55x


@register
class SSpaceDampAmp:
    name = "sspace_damp_amp"
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"
    extract = SSpace.extract  # share path: state has U_r, sqrtS, dS_hat, Vh_r (+ optional b)

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d_in"],     # unused; kept for hook signature
        y: Float[Tensor, "b s d_out"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: SSpaceDampAmpC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = shared["U_r"].to(y)
        dS_hat = stacked["dS"][0].to(y)                                  # [r]; supports_multi=False, take k=0
        y_eff  = y - shared["b"].to(y) if "b" in shared else y

        proj_r = y_eff @ U_r                                            # [b, s, r]
        log_scale = (cfg.coeff * dS_hat).clamp(-cfg.clamp_max, cfg.clamp_max)
        scale = log_scale.exp() - 1                                     # [r]
        delta_y = (proj_r * scale) @ U_r.T                              # [b, s, d_out]
        return y + delta_y
