"""Cosine-gated mean-difference steering (CAST-style).

Same `v_L` as mean_diff, but only apply when the residual already aligns with
the steering direction (or its opposite, depending on sign):

$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L \\cdot \\mathbb{1}[\\cos(h, \\hat{v}_L) > \\tau]$$

Intuition: don't steer when the model is already on-topic (cos high, no need)
-- wait, that's backward. CAST gates so the steering only fires when the prompt
**matches** a target concept. Here we gate on `cos > τ` which means "concept is
already weakly present, amplify it." Flip the sign of `tau` to get the opposite
gating.

Refs:
  - Lee et al. 2024 (CAST) https://arxiv.org/abs/2409.05907
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register
from .mean_diff import MeanDiff, MeanDiffConfig  # noqa: F401  (reuse extract)


@register_config
@dataclass
class CosineGatedConfig(SteeringConfig):
    method: str = "cosine_gated"
    tau: float = 0.0
    normalize: bool = True


@register
class CosineGated:
    name = "cosine_gated"

    @staticmethod
    def extract(pos_acts, neg_acts, cfg: CosineGatedConfig):
        # delegate to mean_diff
        md_cfg = MeanDiffConfig(
            method="mean_diff", layers=cfg.layers, coeff=cfg.coeff,
            target=cfg.target, dtype=cfg.dtype, seed=cfg.seed, normalize=cfg.normalize,
        )
        return MeanDiff.extract(pos_acts, neg_acts, md_cfg)

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: CosineGatedConfig,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h.dtype).to(h.device)
        v_norm = v / (v.norm() + 1e-8)
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        cos = (h_norm * v_norm).sum(dim=-1, keepdim=True)  # [b, s, 1]
        gate = (cos > cfg.tau).to(h.dtype)
        return h + gate * cfg.coeff * v
