"""Cosine-gated mean-difference steering (CAST-inspired soft self-gate).

Same `v_L` as mean_diff, but the update is gated by how aligned the residual
already is with the steering direction. We use **|cos|** because we don't care
about sign (steering can flip a feature; what matters is overlap), and a **soft
gate** (relu shifted by tau) instead of CAST's binary condition.

$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L \\cdot \\max(0, |\\cos(h, \\hat{v}_L)| - \\tau)$$

When `tau=0`, gate ∈ [0, 1] = |cos|, full proportional. When `tau=0.1`, only
fires for tokens with overlap > 0.1.

Refs:
    - Inspired by CAST / conditional activation steering. This is not IBM CAST:
        it uses the same vector for behavior and condition and a soft per-token gate.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class CosineGatedC(SteeringConfig):
    method: str = "cosine_gated"
    tau: float = 0.0  # offset on |cos|; 0 = pure proportional |cos| gate
    normalize: bool = True


@register
class CosineGated:
    name = "cosine_gated"

    @staticmethod
    def extract(pos_acts, neg_acts, cfg: CosineGatedC):
        out = {}
        for li in pos_acts:
            v = pos_acts[li].float().mean(0) - neg_acts[li].float().mean(0)
            if cfg.normalize:
                v = v / (v.norm() + ε)

            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: CosineGatedC,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(y)

        v_norm = v / (v.norm() + ε)
        y_norm = y / (y.norm(dim=-1, keepdim=True) + ε)

        cos  = (y_norm * v_norm).sum(dim=-1, keepdim=True)
        gate = torch.relu(cos.abs() - cfg.tau)              # soft, sign-agnostic

        return y + gate * cfg.coeff * v
