"""PCA steering (RepE/LAT-inspired, vgel pca_diff-like).

For each layer L, compute PCA on the **paired differences** `h^+ - h^-`. Take
the top principal component as the steering direction.

$$D_L = H^+_L - H^-_L \\in \\mathbb{R}^{n\\times d}$$
$$U, S, V^T = \\text{SVD}(D_L - \\bar{D}_L)$$
$$\\text{sign}_L = \\text{sign}(\\bar{D}_L \\cdot V_{:,0})$$
$$v_L = V_{:,0} \\cdot \\text{sign}_L$$

Sign-fixed by aligning the sign-ambiguous top PC to the MEAN paired-difference
(the persona contrast hs), so +coeff always moves toward the positive pole. This
matches repeng's orient-to-positive-class rule (it projects the uncentered
hiddens and flips if pos-mean < neg-mean) and AntiPaSTO's sign(mean(diff_S)).
(Claude 2026-07-15) The prior sign rule voted on CENTERED projections, whose mean
is zero by construction, so it measured the variance cloud's skew rather than
concept polarity and flipped the steering direction at random -- see
AntiPaSTO_concepts/README.md:577-582. This is a lightweight control-vector
baseline, not the full Zou et al. LAT reader: it omits per-diff normalization,
label-based sign selection, and train-mean recentering for reading scores.

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
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")

            diffs    = (pos_acts[li] - neg_acts[li]).float()
            centered = diffs - diffs.mean(0, keepdim=True)

            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            v = Vh[: cfg.n_components]

            # orient each PC to the mean paired-diff (hs), not the centered-cloud skew
            mean_proj = diffs.mean(0) @ v.T                 # [n_components]
            v = v * torch.sign(mean_proj + ε)[:, None]

            if cfg.n_components == 1:
                v = v.squeeze(0)
                if cfg.normalize:
                    v = v / (v.norm() + ε)
                out[li] = {"shared": {"v": v}, "stacked": {}}
            else:
                if cfg.normalize:
                    v = v / (v.norm(dim=1, keepdim=True) + ε)
                out[li] = {"shared": {"V": v}, "stacked": {}}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: PCAC,
    ) -> Float[Tensor, "b s d"]:
        if "v" in shared:
            v = shared["v"].to(y)
            return y + cfg.coeff * v
        V = shared["V"].to(y)
        return y + cfg.coeff * V.sum(0)
