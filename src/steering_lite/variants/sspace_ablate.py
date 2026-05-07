"""Weight-SVD ablation in S-space.

Companion to `sspace`: same extract path (SVD a Linear's weight, recover
whitened S-space coordinates from the output via `(y - b) @ U_r / sqrt(S)`,
compute contrastive mean-diff direction `d_S_hat`), but `apply` projects
`d_S_hat` *out of* `x_S` instead of nudging along it. The output
perturbation is then lifted back via `(delta_S * sqrt(S)) @ U_r^T`.

Math (per token, k=1):

    x_S      = (y - b) @ U_r / sqrt(S)         # whitened S-space coords
    proj     = (x_S . d_S_hat)                 # scalar component along contrastive dir
    delta_S  = -proj * d_S_hat (+ alpha * d_S_hat)   # ablation + optional nudge
    delta_y  = (delta_S * sqrt(S)) @ U_r^T     # lift back to out-space
    y'       = y + delta_y

Multi-round (k stacked):
  Naive sum of independent rank-1 ablations is *wrong* if directions overlap
  (the shared component would be subtracted multiple times). We orthonormalize
  the stack via QR, then project x_S onto that subspace and ablate the
  projection in one shot. For k=1 reduces exactly to the formula above.

Gating: none. The projection magnitude `(x_S . d_S_hat)` IS the input-dependent
strength, so a separate cosine gate would be redundant. Tokens with no
contrastive component get a near-zero ablation by construction.

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
    extract = SSpace.extract  # shared: U_r, sqrtS, Vh_r (+ optional b); stacked: dS [k,r]

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d_in"],     # unused; kept for hook signature
        y: Float[Tensor, "b s d_out"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: SSpaceAblateC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = shared["U_r"].to(y)
        sqrtS  = shared["sqrtS"].to(y)
        dS_raw = stacked["dS"].to(y)                          # [k, r], rows = alpha_i * dS_hat_i
        y_eff  = y - shared["b"].to(y) if "b" in shared else y

        xS = (y_eff @ U_r) / sqrtS                            # [b, s, r]
        # QR-orthonormalize for clean k-dim subspace projection (k=1 reduces to dS_hat)
        Q, _ = torch.linalg.qr(dS_raw.T.float())              # float32: geqrf_cuda unsupported for bf16
        Q = Q.to(y)
        proj_S = (xS @ Q) @ Q.T                               # [b, s, r]
        delta_S = -proj_S
        if cfg.coeff != 0.0:
            delta_S = delta_S + cfg.coeff * dS_raw.sum(0)     # weighted-sum nudge
        delta_y = (delta_S * sqrtS) @ U_r.T                   # [b, s, d_out]
        return y + delta_y
