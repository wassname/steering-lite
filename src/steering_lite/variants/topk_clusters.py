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

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class TopKClustersC(SteeringConfig):
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
        Xn = X / (X.norm(dim=1, keepdim=True) + ε)
        Cn = C / (C.norm(dim=1, keepdim=True) + ε)
        sim = einsum(Xn, Cn, "n d, k d -> n k")
        assign = sim.argmax(dim=1)  # [n]
        counts = torch.bincount(assign, minlength=k)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if len(empty) > 0:
            # reinitialize empty clusters from random data points
            rand_idx = torch.randperm(n, generator=g)[: len(empty)]
            C[empty] = X[rand_idx].clone()
            continue
        new_C = torch.stack([X[assign == j].mean(0) for j in range(k)])
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
        cfg: TopKClustersC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            if cfg.k > pos_acts[li].shape[0]:
                raise ValueError(f"k={cfg.k} exceeds n={pos_acts[li].shape[0]}")

            diffs = (pos_acts[li] - neg_acts[li]).float()
            C     = _kmeans(diffs, k=cfg.k, n_iters=cfg.n_iters, seed=cfg.seed)

            # Per-centroid sign canon: cosine k-means can converge to centroids
            # anti-aligned with the global honesty axis (a syntactic-noise mode).
            # Eval-time flip can't fix one centroid alone (it flips all). So
            # project each onto the global mean diff; flip if negative.
            g    = diffs.mean(dim=0)
            proj = einsum(C, g, "k d, d -> k")
            sign = torch.where(proj < 0, -1.0, 1.0).to(C.dtype)
            C    = C * sign.unsqueeze(-1)

            if cfg.normalize:
                C = C / (C.norm(dim=1, keepdim=True) + ε)

            out[li] = {"C": C}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: TopKClustersC,
    ) -> Float[Tensor, "b s d"]:
        C = state["C"].to(y)
        # cos-sim per token; pick best centroid per token
        y_norm = y / (y.norm(dim=-1, keepdim=True) + ε)
        C_norm = C / (C.norm(dim=-1, keepdim=True) + ε)

        sim  = einsum(y_norm, C_norm, "b s d, k d -> b s k")
        pick = sim.argmax(dim=-1)
        v    = C[pick]

        return y + cfg.coeff * v
