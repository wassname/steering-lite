"""PCA steering (RepE-style).

For each layer L, compute PCA on the **paired differences** `h^+ - h^-`. Take
the top principal component as the steering direction.

$$D_L = H^+_L - H^-_L \\in \\mathbb{R}^{n\\times d}$$
$$U, S, V^T = \\text{SVD}(D_L - \\bar{D}_L)$$
$$v_L = V_{:,0} \\cdot \\text{sign}(\\langle V_{:,0}, \\bar{D}_L \\rangle)$$

Sign-flipped so the direction agrees with the average difference (PCA is
sign-ambiguous; the bisector to the mean diff fixes it).

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
            # sign-fix each PC by alignment with the mean diff
            sign = torch.sign((v @ mean.squeeze(0)) + 1e-8)
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
