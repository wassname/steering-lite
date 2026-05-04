"""PCA steering (RepE/LAT-inspired, vgel pca_diff-like).

For each layer L, compute PCA on the **paired differences** `h^+ - h^-`. Take
the top principal component as the steering direction.

$$D_L = H^+_L - H^-_L \\in \\mathbb{R}^{n\\times d}$$
$$U, S, V^T = \\text{SVD}(D_L - \\bar{D}_L)$$
$$\\text{sign}_L = \\text{sign}\\left(\\sum_i \\mathbb{1}[(D_L)_i \\cdot V_{:,0} > 0] - n/2\\right)$$
$$v_L = V_{:,0} \\cdot \\text{sign}_L$$

Sign-fixed by majority vote of paired-diff projections (repeng/vgel style).
This is a lightweight control-vector baseline, not the full Zou et al. LAT
reader: it omits per-diff normalization, label-based sign selection, and
train-mean recentering for reading scores.
PCA is sign-ambiguous; the vote is more robust than alignment-with-the-mean
when paired diffs are heterogeneous (mean can cancel without the vote
changing). If the vote ties exactly, orient the axis so the largest centered
projection is positive.

At runtime, add `coeff * v_L` to the residual.

Refs:
  - Zou et al. 2023 (Representation Engineering) https://arxiv.org/abs/2310.01405
  - vgel/repeng: https://github.com/vgel/repeng
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class PCAC(SteeringConfig):
    method: str = "pca"
    n_components: int = 1
    normalize: bool = True


@register
class PCA:
    name = "pca"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: PCAC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")

            diffs    = (pos_acts[li] - neg_acts[li]).float()
            centered = diffs - diffs.mean(0, keepdim=True)

            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            v = Vh[: cfg.n_components]

            projs         = centered @ v.T
            positive_frac = (projs > 0).float().mean(0)
            majority_sign = torch.where(positive_frac > 0.5,
                               torch.ones(v.shape[0]),
                               -torch.ones(v.shape[0])).to(v)
            strongest_idx  = projs.abs().argmax(dim=0)
            strongest      = projs[strongest_idx, torch.arange(v.shape[0], device=projs.device)]
            strongest_sign = torch.sign(strongest)
            sign           = torch.where(positive_frac == 0.5, strongest_sign, majority_sign)
            v = v * sign[:, None]

            if cfg.n_components == 1:
                v = v.squeeze(0)
                if cfg.normalize:
                    v = v / (v.norm() + ε)
                out[li] = {"v": v}
            else:
                if cfg.normalize:
                    v = v / (v.norm(dim=1, keepdim=True) + ε)
                out[li] = {"V": v}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: PCAC,
    ) -> Float[Tensor, "b s d"]:
        if "v" in state:
            v = state["v"].to(y)
            return y + cfg.coeff * v
        # multi-component: sum top-k directions equally
        V = state["V"].to(y)
        return y + cfg.coeff * V.sum(0)
