"""Cosine-gated mean-difference steering (CAST-style, soft + sign-agnostic).

Same `v_L` as mean_diff, but the update is gated by how aligned the residual
already is with the steering direction. We use **|cos|** because we don't care
about sign (steering can flip a feature; what matters is overlap), and a **soft
gate** (relu shifted by tau) instead of a hard 0/1 — so a hard threshold can't
silently drop the whole intervention to zero on a different model.

$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L \\cdot \\max(0, |\\cos(h, \\hat{v}_L)| - \\tau)$$

When `tau=0`, gate ∈ [0, 1] = |cos|, full proportional. When `tau=0.1`, only
fires for tokens with overlap > 0.1.

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
    tau: float = 0.0  # offset on |cos|; 0 = pure proportional |cos| gate
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
        gate = torch.relu(cos.abs() - cfg.tau)              # soft, sign-agnostic
        return h + gate * cfg.coeff * v
