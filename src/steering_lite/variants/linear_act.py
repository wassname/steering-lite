"""Linear-AcT: coordinate-wise affine activation transport.

Rodriguez et al. 2025 ICLR, "Controlling Language and Diffusion Models by
Transporting Activations" https://openreview.net/forum?id=l2zFn6TIQi

Linear-AcT fits one univariate affine OT map per activation coordinate. For
source samples $a_i$ and target samples $b_i$, sort each coordinate, centre the
sorted values, and fit

$$T_j(x_j) = \\omega_j x_j + \\beta_j$$

with

$$\\omega_j = \\frac{\\sum_i \\tilde a_{(i),j} \\tilde b_{(i),j}}{\\sum_i \\tilde a_{(i),j}^2}, \\quad
\\beta_j = m_{b,j} - \\omega_j m_{a,j}.$$

Apply strength $\\alpha$ by interpolation:

$$h \\leftarrow (1-\\alpha)h + \\alpha T(h).$$

This is the paper's core coordinate-wise `Linear-AcT` map, not the multivariate
Gaussian/Bures OT map. It omits optional support masking and layerwise map
estimation from the full pipeline.
"""
from dataclasses import dataclass
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class LinearAcTConfig(SteeringConfig):
    method: str = "linear_act"


@register
class LinearAcT:
    name = "linear_act"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: LinearAcTConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            X_a = neg_acts[li].float()  # source
            X_b = pos_acts[li].float()  # target
            if X_a.shape[0] < 2:
                raise ValueError("Linear-AcT needs at least two samples")
            a_sorted = X_a.sort(dim=0).values
            b_sorted = X_b.sort(dim=0).values
            m_a = a_sorted.mean(dim=0)
            m_b = b_sorted.mean(dim=0)
            a_tilde = a_sorted - m_a
            b_tilde = b_sorted - m_b
            denom = (a_tilde ** 2).sum(dim=0)
            omega = (a_tilde * b_tilde).sum(dim=0) / denom
            beta = m_b - omega * m_a
            out[li] = {"omega": omega, "beta": beta}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: LinearAcTConfig,
    ) -> Float[Tensor, "b s d"]:
        omega = rearrange(state["omega"].to(h.dtype).to(h.device), "d -> 1 1 d")
        beta = rearrange(state["beta"].to(h.dtype).to(h.device), "d -> 1 1 d")
        h_new = h * omega + beta
        return (1 - cfg.coeff) * h + cfg.coeff * h_new
