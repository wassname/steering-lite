"""Angular Steering: fixed-plane rotation in activation space.

Vu & Nguyen 2025 NeurIPS https://arxiv.org/abs/2510.26243

Construct a fixed orthonormal steering plane $(b_1, b_2)$ per layer:
$b_1$ is the normalized mean contrast direction and $b_2$ is the first PC of
normalized pairwise contrast directions, orthogonalized against $b_1$.
Set the projection's in-plane angle to $\\theta$ while preserving its in-plane
norm (the paper's fixed-plane target-angle form):

$$h' = h - P h + \\lVert P h \\rVert (\\cos\\theta \\, b_1 + \\sin\\theta \\, b_2).$$

The orthogonal component is untouched and the in-plane norm is preserved, so
total residual norm is preserved. `cfg.coeff` is $\\theta$ in radians.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class AngularSteeringC(SteeringConfig):
    method: str = "angular_steering"
    # cfg.coeff is interpreted as rotation angle theta (radians).


@register
class AngularSteering:
    name = "angular_steering"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: AngularSteeringC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")

            diffs = (pos_acts[li].float() - neg_acts[li].float())
            dirs  = diffs / (diffs.norm(dim=1, keepdim=True) + ε)

            b1 = diffs.mean(0)
            b1 = b1 / (b1.norm() + ε)

            _, _, Vh = torch.linalg.svd(dirs - dirs.mean(0, keepdim=True), full_matrices=False)
            b2 = Vh[0] - (Vh[0] @ b1) * b1
            b2 = b2 / (b2.norm() + ε)

            out[li] = {"b1": b1, "b2": b2}
        return out

    @staticmethod
    def apply(
        block,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: AngularSteeringC,
    ) -> Float[Tensor, "b s d"]:
        b1 = state["b1"].to(y)
        b2 = state["b2"].to(y)

        c1 = (y * b1).sum(dim=-1, keepdim=True)
        c2 = (y * b2).sum(dim=-1, keepdim=True)
        y_plane    = c1 * b1 + c2 * b2
        plane_norm = y_plane.norm(dim=-1, keepdim=True)

        theta      = torch.tensor(float(cfg.coeff), dtype=y.dtype, device=y.device)
        target_dir = torch.cos(theta) * b1 + torch.sin(theta) * b2

        return y - y_plane + plane_norm * target_dir
