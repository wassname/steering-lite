"""PCA steering (RepE-style).

For each layer L, compute PCA on the **paired differences** `h^+ - h^-`. Take
the top principal component as the steering direction.

$$D_L = H^+_L - H^-_L \\in \\mathbb{R}^{n\\times d}$$
$$U, S, V^T = \\text{SVD}(D_L - \\bar{D}_L)$$
$$\\text{sign}_L = \\text{sign}\\left(\\sum_i \\mathbb{1}[(D_L)_i \\cdot V_{:,0} > 0] - n/2\\right)$$
$$v_L = V_{:,0} \\cdot \\text{sign}_L$$

Sign-fixed by majority vote of paired-diff projections (repeng/vgel style).
PCA is sign-ambiguous; the vote is more robust than alignment-with-the-mean
when paired diffs are heterogeneous (mean can cancel without the vote
changing). If the vote is close to 50/50 the principal axis isn't well
defined and `mean_diff` is the better method to begin with.

At runtime, add `coeff * v_L` to the residual.

Refs:
  - Zou et al. 2023 (Representation Engineering) https://arxiv.org/abs/2310.01405
  - vgel/repeng: https://github.com/vgel/repeng
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class PCAConfig(SteeringConfig):
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
        cfg: PCAConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            # paired diffs assume aligned ordering of pos/neg prompts
            n = min(pos_acts[li].shape[0], neg_acts[li].shape[0])
            diffs = (pos_acts[li][:n] - neg_acts[li][:n]).float()  # [n, d]
            mean = diffs.mean(0, keepdim=True)
            centered = diffs - mean
            # SVD on centered diffs; right singular vectors are PC directions
            _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
            v = Vh[: cfg.n_components]  # [k, d]
            # sign-fix each PC by majority vote across paired diffs (repeng style):
            # project each diff onto the PC and check whether more land on the
            # positive or negative side. More robust to outliers than aligning
            # with the (small) mean diff.
            projs = diffs @ v.T  # [n, k]
            sign = torch.where((projs > 0).float().mean(0) >= 0.5,
                               torch.ones(v.shape[0]),
                               -torch.ones(v.shape[0])).to(v)
            v = v * sign[:, None]
            if cfg.n_components == 1:
                v = v.squeeze(0)
                if cfg.normalize:
                    v = v / (v.norm() + 1e-8)
                out[li] = {"v": v}
            else:
                if cfg.normalize:
                    v = v / (v.norm(dim=1, keepdim=True) + 1e-8)
                out[li] = {"V": v}  # [k, d]
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: PCAConfig,
    ) -> Float[Tensor, "b s d"]:
        if "v" in state:
            v = state["v"].to(h.dtype).to(h.device)
            return h + cfg.coeff * v
        # multi-component: sum top-k directions equally
        V = state["V"].to(h.dtype).to(h.device)  # [k, d]
        return h + cfg.coeff * V.sum(0)
