"""Mean-difference steering (CAA / ActAdd).

For each selected layer L, compute the mean difference between positive and
negative last-token hidden states:

$$v_L = \\text{mean}(h^+_L) - \\text{mean}(h^-_L), \\quad \\hat{v}_L = v_L / \\|v_L\\|$$

At runtime, add `coeff * v_hat` to every token's residual at that block:

$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L$$

Refs:
  - Panickssery et al. 2023 https://arxiv.org/abs/2312.06681 (CAA)
  - Turner et al. 2023 https://arxiv.org/abs/2308.10248 (ActAdd)
  - nrimsky/CAA: https://github.com/nrimsky/CAA
"""
from dataclasses import dataclass
from einops import einsum  # noqa: F401  (kept for symmetry; not needed for this method)
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class MeanDiffConfig(SteeringConfig):
    method: str = "mean_diff"
    normalize: bool = True


@register
class MeanDiff:
    name = "mean_diff"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: MeanDiffConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            v = pos_acts[li].float().mean(0) - neg_acts[li].float().mean(0)
            if cfg.normalize:
                v = v / (v.norm() + 1e-8)
            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: MeanDiffConfig,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h.dtype).to(h.device)
        return h + cfg.coeff * v
