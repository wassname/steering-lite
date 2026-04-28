"""Spherical steering (slerp on residual hypersphere).

Treat the residual as a point on the (d-1)-sphere and rotate it toward a target
direction `v` by angle `coeff` (radians). Slerp preserves norm exactly:

$$h_{\\text{rot}} = \\text{slerp}(\\hat{h}, \\hat{v}, \\alpha) \\cdot \\|h\\|$$

where slerp is

$$\\text{slerp}(a, b, t) = \\frac{\\sin((1-t)\\Omega)}{\\sin\\Omega} a + \\frac{\\sin(t\\Omega)}{\\sin\\Omega} b, \\quad \\Omega = \\arccos(a \\cdot b)$$

For numerical stability when `a` and `b` are nearly parallel, we fall back to
linear interpolation.

Refs:
  - chili-lab/Spherical-Steering https://github.com/chili-lab/Spherical-Steering
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class SphericalConfig(SteeringConfig):
    method: str = "spherical"
    # coeff is interpreted here as the slerp t in [0, 1] (or beyond, extrapolation).
    # Default 0.1 = small rotation toward v.


@register
class Spherical:
    name = "spherical"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: SphericalConfig,
    ) -> dict[int, dict[str, Tensor]]:
        # target direction = unit mean diff (same as mean_diff)
        out = {}
        for li in pos_acts:
            v = pos_acts[li].float().mean(0) - neg_acts[li].float().mean(0)
            v = v / (v.norm() + 1e-8)
            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: SphericalConfig,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h.dtype).to(h.device)  # unit
        norm = h.norm(dim=-1, keepdim=True)  # [b, s, 1]
        h_hat = h / (norm + 1e-8)  # [b, s, d]
        cos = (h_hat * v).sum(dim=-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
        omega = torch.arccos(cos)  # [b, s, 1]
        sin_omega = torch.sin(omega)
        t = cfg.coeff
        # slerp; broadcast scalars
        a = torch.sin((1 - t) * omega) / sin_omega
        b = torch.sin(t * omega) / sin_omega
        rot = a * h_hat + b * v
        # fallback to lerp where omega ~ 0
        near_parallel = sin_omega.abs() < 1e-4
        lerp = (1 - t) * h_hat + t * v
        rot = torch.where(near_parallel, lerp, rot)
        return rot * norm
