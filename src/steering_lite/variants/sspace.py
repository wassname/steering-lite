"""S-space (SVD subspace) steering.

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

Refs:
  - wassname/ssteer-eval-aware https://github.com/wassname/ssteer-eval-aware
  - Subspace taxonomy: docs/AntiPaSTO_concepts/docs/steering_methods.qmd
"""
from dataclasses import dataclass
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class SSpaceConfig(SteeringConfig):
    method: str = "sspace"
    r: int = 4
    project_at_runtime: bool = False  # if True, project residual onto subspace and scale
    normalize: bool = True


@register
class SSpace:
    name = "sspace"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: SSpaceConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            n = min(pos_acts[li].shape[0], neg_acts[li].shape[0])
            diffs = (pos_acts[li][:n] - neg_acts[li][:n]).float()  # [n, d]
            mean_diff = diffs.mean(0)  # [d]
            r = min(cfg.r, diffs.shape[0], diffs.shape[1])
            # right singular vectors -> directions in d-space
            _, _, Vh = torch.linalg.svd(diffs, full_matrices=False)
            V = Vh[:r].T.contiguous()  # [d, r]; .T makes a view, safetensors needs contiguous
            # project mean diff onto subspace
            v = V @ (V.T @ mean_diff)  # [d]
            if cfg.normalize:
                v = v / (v.norm() + 1e-8)
            out[li] = {"v": v, "V": V}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: SSpaceConfig,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h.dtype).to(h.device)
        if not cfg.project_at_runtime:
            return h + cfg.coeff * v
        # alt mode: project residual onto subspace and scale that part
        V = state["V"].to(h.dtype).to(h.device)  # [d, r]
        proj = einsum(h, V, "b s d, d r -> b s r")
        h_proj = einsum(proj, V, "b s r, d r -> b s d")
        return h + cfg.coeff * h_proj
