"""Random-direction null steering (placebo).

A per-layer random unit direction, independent of the contrastive data, added
like mean_diff / CAA. Calibrated to the same iso-KL dose as every real method,
so its selectivity score is the floor from an arbitrary perturbation of equal KL
magnitude. If a real method doesn't clear this null, its "steering" is generic
disruption (any push of this size moves the axis), not a specific-axis move.

Seeded per (cfg.seed, layer) so directions are reproducible and independent
across layers; vary cfg.seed to draw the null distribution (a null is a
distribution, not one draw). (Claude, not wassname)
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class RandomC(SteeringConfig):
    method: str = "random"
    normalize: bool = True


@register
class Random:
    name = "random"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: RandomC,
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        # Ignore the contrastive content; only the shape (d) and layer keys are used.
        out = {}
        for li in pos_acts:
            ref = pos_acts[li]
            g = torch.Generator().manual_seed(cfg.seed * 10_000 + li)
            v = torch.randn(ref.shape[-1], generator=g).to(ref).float()
            if cfg.normalize:
                v = v / (v.norm() + ε)
            out[li] = {"shared": {}, "stacked": {"v": v.unsqueeze(0)}}  # [1, d]
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: RandomC,
    ) -> Float[Tensor, "b s d"]:
        v_stack = stacked["v"].to(y)            # [k, d]
        return y + cfg.coeff * v_stack.sum(dim=0)
