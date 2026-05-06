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

Multi-round (k stacked):
  Composition of multiplicative effects = multiplication of scales = ADDITION
  of log-scales. So accumulating k rounds is the row-wise sum of stacked dS:

      log_scale = c * Σ_i (alpha_i * d_S_hat_i)         # [r]   (= c * stacked.sum(0))
      scale     = clamp_exp(log_scale) - 1
      delta_y   = (proj_r * scale) @ U_r^T

  For k=1 reduces to the single-round formula above.

Properties:
  - Monotone in c (larger |c| -> stronger effect).
  - S_eff = σ * exp(...) > 0 always; never sign-flips a mode.
  - At c=0 the steering is exactly identity (scale=0).
  - Non-selected modes contribute 0 (they are absent from U_r).
  - No cosine gate: the per-mode multiplier IS the gating signal (high-|dS_hat_i|
    modes get more amplification; low-|dS_hat_i| modes are nearly identity).

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
    extract = SSpace.extract  # shared: U_r, sqrtS, Vh_r (+ optional b); stacked: dS [k,r]

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
        dS_raw = stacked["dS"].to(y)                                    # [k, r]
        y_eff  = y - shared["b"].to(y) if "b" in shared else y

        proj_r = y_eff @ U_r                                            # [b, s, r]
        # multi-round composition: log-scales add (multiplicative effects compose)
        log_scale = (cfg.coeff * dS_raw.sum(0)).clamp(-cfg.clamp_max, cfg.clamp_max)
        scale = log_scale.exp() - 1                                     # [r]
        delta_y = (proj_r * scale) @ U_r.T                              # [b, s, d_out]
        return y + delta_y
