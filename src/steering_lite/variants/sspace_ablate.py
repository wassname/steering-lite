"""Weight-SVD ablation in S-space.

Companion to `sspace`: same extract path (SVD a Linear's weight, recover
whitened S-space coordinates from the output via `(y - b) @ U_r / sqrt(S)`,
compute contrastive mean-diff direction `d_S_hat`), but `apply` projects
`d_S_hat` *out of* `x_S` instead of nudging along it. The output
perturbation is then lifted back via `(delta_S * sqrt(S)) @ U_r^T`.

Math (per token):

    x_S      = (y - b) @ U_r / sqrt(S)         # whitened S-space coords
    proj     = (x_S . d_S_hat)                 # scalar component along contrastive dir
    delta_S  = -proj * d_S_hat (+ alpha * d_S_hat)   # ablation + optional nudge
    delta_y  = (delta_S * sqrt(S)) @ U_r^T     # lift back to out-space
    y'       = y + delta_y

When `coeff = 0` this is pure projection-out -- a 1-D ablation in the
weight-SVD subspace. Norm shrinkage in S-space is `|x_S . d_S_hat|`, an
informative diagnostic: if it is near zero the contrastive direction wasn't
present and the intervention is a no-op.

Compare to:
- `directional_ablation.py`: ablation in full residual d-space. Removes the
  direction from h regardless of how aligned the model's actual computation
  is with that direction.
- `sspace.py` (cosine-gated additive): pushes along d_S_hat with a strength
  proportional to S-space alignment.

Together with `sspace`, this gives an additive vs subtractive comparison at
the same hook site / extract path.
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
class SSpaceAblateC(SteeringConfig):
    method: str = "sspace_ablate"
    r: int = -1  # -1 = full rank; else top-r modes by |d_S|
    coeff: float = 0.0  # 0 = pure projection-out; !=0 adds alpha * d_S_hat after ablation


@register
class SSpaceAblate:
    name = "sspace_ablate"
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"
    extract = SSpace.extract  # share path: state has U_r, sqrtS, dS_hat (+optional b)

    @staticmethod
    def apply(
        block,
        x: Float[Tensor, "b s d_in"],     # unused; kept for hook signature
        y: Float[Tensor, "b s d_out"],
        state: dict[str, Tensor],
        cfg: SSpaceAblateC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = state["U_r"].to(y)
        sqrtS  = state["sqrtS"].to(y)
        dS_hat = state["dS_hat"].to(y)
        y_eff = y - state["b"].to(y) if "b" in state else y

        xS = (y_eff @ U_r) / sqrtS                            # [b, s, r]
        proj = (xS * dS_hat).sum(dim=-1, keepdim=True)        # [b, s, 1]
        delta_S = -proj * dS_hat                              # ablate, [b, s, r]
        if cfg.coeff != 0.0:
            delta_S = delta_S + cfg.coeff * dS_hat            # broadcast nudge
        delta_y = (delta_S * sqrtS) @ U_r.T                   # [b, s, d_out]
        return y + delta_y
