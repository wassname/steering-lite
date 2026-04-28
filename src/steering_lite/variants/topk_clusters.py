"""Top-k cluster steering.

Cosine-assignment k-means on the paired diffs. At runtime, pick the centroid most aligned with
the current residual (max cosine similarity) and add it. Idea: different
prompts may need different steering directions; clusters discover the modes.

$$\\{c_1, ..., c_k\\} = \\text{cosine-kmeans}_k(H^+ - H^-)$$
$$h \\leftarrow h + \\alpha \\cdot c_{j^*}, \\quad j^* = \\arg\\max_j \\cos(h, c_j)$$

Folk / novel-ish baseline. Closest neighbours are CHaRS-style cluster-aware
steering and SVF/KNN local steering, but this exact cosine-kmeans router is an
internal baseline.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor
from einops import einsum

from ..config import SteeringConfig, register_config
from ..method import register


@register_config
@dataclass
class TopKClustersConfig(SteeringConfig):
    method: str = "topk_clusters"
    k: int = 4
    n_iters: int = 20
    normalize: bool = True


def _kmeans(X: Tensor, k: int, n_iters: int, seed: int) -> Tensor:
    """Return centroids `[k, d]`. Pure-torch, no sklearn dep."""
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    C = X[init_idx].clone()
    for _ in range(n_iters):
        # cosine assignment (for direction-ish data)
        Xn = X / (X.norm(dim=1, keepdim=True) + 1e-8)
        Cn = C / (C.norm(dim=1, keepdim=True) + 1e-8)
        sim = einsum(Xn, Cn, "n d, k d -> n k")
        assign = sim.argmax(dim=1)  # [n]
        new_C = torch.stack([
            X[assign == j].mean(0) if (assign == j).any() else C[j]
            for j in range(k)
        ])
        if torch.allclose(new_C, C, atol=1e-6):
            break
        C = new_C
    return C


@register
class TopKClusters:
    name = "topk_clusters"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: TopKClustersConfig,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            diffs = (pos_acts[li] - neg_acts[li]).float()
            if cfg.k > diffs.shape[0]:
                raise ValueError(f"k={cfg.k} exceeds n={diffs.shape[0]}")
            k = cfg.k
            C = _kmeans(diffs, k=k, n_iters=cfg.n_iters, seed=cfg.seed)  # [k, d]
            if cfg.normalize:
                C = C / (C.norm(dim=1, keepdim=True) + 1e-8)
            out[li] = {"C": C}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: TopKClustersConfig,
    ) -> Float[Tensor, "b s d"]:
        C = state["C"].to(h.dtype).to(h.device)  # [k, d]
        # cos-sim per token; pick best centroid per token
        h_norm = h / (h.norm(dim=-1, keepdim=True) + 1e-8)
        C_norm = C / (C.norm(dim=-1, keepdim=True) + 1e-8)
        sim = einsum(h_norm, C_norm, "b s d, k d -> b s k")
        pick = sim.argmax(dim=-1)  # [b, s]
        v = C[pick]  # [b, s, d]
        return h + cfg.coeff * v
