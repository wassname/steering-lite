"""Ungated spherical steering core (slerp on residual hypersphere).

Treat the residual as a point on the (d-1)-sphere and rotate it toward a target
direction `v` by slerp fraction `coeff`. Slerp preserves norm except at the
antipodal degeneracy where the geodesic is not unique:

$$h_{\\text{rot}} = \\text{slerp}(\\hat{h}, \\hat{v}, \\alpha) \\cdot \\|h\\|$$

where slerp is

$$\\text{slerp}(a, b, t) = \\frac{\\sin((1-t)\\Omega)}{\\sin\\Omega} a + \\frac{\\sin(t\\Omega)}{\\sin\\Omega} b, \\quad \\Omega = \\arccos(a \\cdot b)$$

This is the fixed-t, ungated core of Spherical Steering. It omits the paper's
vMF confidence gate (`kappa`, `alpha`, `beta`).

Refs:
    - Spherical Steering https://arxiv.org/abs/2602.08169
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
class SphericalC(SteeringConfig):
    method: str = "spherical"
    # coeff is the slerp fraction t in [0, 1] (or beyond, extrapolation).


@register
class Spherical:
    name = "spherical"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: SphericalC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            v = pos_acts[li].float().mean(0) - neg_acts[li].float().mean(0)
            v = v / v.norm()

            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: SphericalC,
    ) -> Float[Tensor, "b s d"]:
        v     = state["v"].to(h)  # unit
        norm  = h.norm(dim=-1, keepdim=True)
        h_hat = h / norm

        cos       = (h_hat * v).sum(dim=-1, keepdim=True)
        omega     = torch.arccos(cos)
        sin_omega = torch.sin(omega)

        t = cfg.coeff
        a = torch.sin((1 - t) * omega) / sin_omega
        b = torch.sin(t * omega) / sin_omega
        rot = a * h_hat + b * v

        return rot * norm
