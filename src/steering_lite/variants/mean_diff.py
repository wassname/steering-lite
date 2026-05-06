"""Mean-difference steering (CAA / ActAdd).

For each selected layer L, compute the mean difference between positive and
negative last-token hidden states:

$$v_L = \\text{mean}(h^+_L) - \\text{mean}(h^-_L), \\quad \\hat{v}_L = v_L / \\|v_L\\|$$

At runtime, add `coeff * sum_i v_i` to every token's residual at that block:

$$h \\leftarrow h + \\alpha \\cdot \\sum_i v_i$$

Linear method: stacked rows can be summed at apply time, equivalent to
applying each round sequentially.

Refs:
  - Panickssery 2023 (CAA) https://arxiv.org/abs/2312.06681
  - Turner 2023 (ActAdd) https://arxiv.org/abs/2308.10248
  - Jorgensen 2024 (Mean-Centring) https://arxiv.org/abs/2312.03813
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class MeanDiffC(SteeringConfig):
    method: str = "mean_diff"
    normalize: bool = True
    subtract_corpus_mean: bool = False


@register
class MeanDiff:
    name = "mean_diff"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: MeanDiffC,
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        out = {}
        for li in pos_acts:
            p = pos_acts[li].float()
            n = neg_acts[li].float()

            if cfg.subtract_corpus_mean:
                mu = torch.cat([p, n], dim=0).mean(0)
                v = p.mean(0) - mu
            else:
                v = p.mean(0) - n.mean(0)

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
        cfg: MeanDiffC,
    ) -> Float[Tensor, "b s d"]:
        v_stack = stacked["v"].to(y)            # [k, d]
        return y + cfg.coeff * v_stack.sum(dim=0)
