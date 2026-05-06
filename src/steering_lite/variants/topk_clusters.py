"""Top-k cluster steering.

Cosine-assignment k-means on the paired diffs. At runtime, pick the centroid
most aligned with the current residual (max cosine similarity) and add it.

$$\\{c_1, ..., c_k\\} = \\text{cosine-kmeans}_k(H^+ - H^-)$$
$$h \\leftarrow h + \\alpha \\cdot c_{j^*}, \\quad j^* = \\arg\\max_j \\cos(h, c_j)$$

Multi-round:

  Stacked tensor `C: [k_rounds, n_clusters, d]`. Each round runs its OWN
  argmax routing (one centroid pick per token per round) then sums deltas.
  This preserves per-round routing semantics -- different rounds may route
  the same token to different cluster sets, which is exactly the point of
  iterating: round 2 picks up modes round 1 missed under the perturbed
  residual.
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
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    init_idx = torch.randperm(n, generator=g)[:k]
    C = X[init_idx].clone()
    for _ in range(n_iters):
        Xn = X / (X.norm(dim=1, keepdim=True) + ε)
        Cn = C / (C.norm(dim=1, keepdim=True) + ε)
        sim = einsum(Xn, Cn, "n d, k d -> n k")
        assign = sim.argmax(dim=1)
        counts = torch.bincount(assign, minlength=k)
        empty = (counts == 0).nonzero(as_tuple=True)[0]
        if len(empty) > 0:
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
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            if cfg.k > pos_acts[li].shape[0]:
                raise ValueError(f"k={cfg.k} exceeds n={pos_acts[li].shape[0]}")

            diffs = (pos_acts[li] - neg_acts[li]).float()
            C     = _kmeans(diffs, k=cfg.k, n_iters=cfg.n_iters, seed=cfg.seed)

            g    = diffs.mean(dim=0)
            proj = einsum(C, g, "k d, d -> k")
            sign = torch.where(proj < 0, -1.0, 1.0).to(C.dtype)
            C    = C * sign.unsqueeze(-1)

            if cfg.normalize:
                C = C / (C.norm(dim=1, keepdim=True) + ε)

            out[li] = {"shared": {}, "stacked": {"C": C.unsqueeze(0)}}  # [1, n_clusters, d]
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: TopKClustersC,
    ) -> Float[Tensor, "b s d"]:
        C_stack = stacked["C"].to(y)            # [k_rounds, n_clusters, d]
        y_norm = y / (y.norm(dim=-1, keepdim=True) + ε)         # [b, s, d]

        delta = torch.zeros_like(y)
        for r_idx in range(C_stack.shape[0]):
            C = C_stack[r_idx]                                  # [n_clusters, d]
            C_norm = C / (C.norm(dim=-1, keepdim=True) + ε)
            sim = einsum(y_norm, C_norm, "b s d, n d -> b s n")
            pick = sim.argmax(dim=-1)                           # [b, s]
            delta = delta + C[pick]
        return y + cfg.coeff * delta
