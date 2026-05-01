"""Activation-diff SVD subspace steering.

Compute SVD of stacked paired diffs at each layer; keep the top-r right
singular vectors as a basis `V \\in \\mathbb{R}^{d\\times r}`. The mean diff is
projected onto this subspace and that's the steering update:

$$D_L = H^+_L - H^-_L$$
$$U, S, V^T = \\text{SVD}(D_L)$$
$$V_r = V_{:,:r}, \\quad P_r = V_r V_r^T$$
$$v_L = P_r \\cdot \\bar{D}_L$$
$$h \\leftarrow h + \\alpha \\cdot \\hat{v}_L$$

Why subspace? The diff vectors span more than one direction; the leading r
captures the task-relevant subspace and rejects per-prompt noise. r=1 reduces
(roughly) to PCA; r>1 averages over the principal subspace.

This is an internal activation-subspace baseline. It is not the weight-SVD
S-space method from ssteer-eval-aware, which steers singular modes of layer
weights via input-dependent perturbations.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class SSpaceC(SteeringConfig):
    method: str = "sspace"
    r: int = 4
    normalize: bool = True


@register
class SSpace:
    name = "sspace"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: SSpaceC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            if cfg.r > min(pos_acts[li].shape[0], pos_acts[li].shape[1]):
                raise ValueError(f"r={cfg.r} exceeds rank bound")

            diffs = (pos_acts[li] - neg_acts[li]).float()
            mu    = diffs.mean(0)

            _, _, Vh = torch.linalg.svd(diffs, full_matrices=False)
            V = Vh[:cfg.r].T.contiguous()  # safetensors needs contiguous

            v = V @ (V.T @ mu)
            if cfg.normalize:
                v = v / (v.norm() + ε)

            out[li] = {"v": v, "V": V}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: SSpaceC,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h)
        return h + cfg.coeff * v
