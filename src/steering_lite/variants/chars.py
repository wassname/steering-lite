"""CHaRS: Concept Heterogeneity-aware Representation Steering.

Abdullaev et al. 2026 https://arxiv.org/abs/2603.02237

Generalises mean_diff to multimodal concepts: instead of one direction per
layer, learn K source clusters $a_i$, K target clusters $b_j$, an OT
coupling $P^*_{ij}$ between them via Sinkhorn, and gate the per-cluster
translations $v_{ij} = b_j - a_i$ by an RBF kernel on distance to source
centroids:

$$\\hat v(x) = \\sum_{ij} \\frac{P^*_{ij} \\, k(x, a_i)}{\\sum_{pq} P^*_{pq} k(x, a_p)} (b_j - a_i)$$

with $k(x, a_i) = \\exp(-\\lVert x - a_i \\rVert^2 / 2\\sigma^2)$ and
$P^*$ from entropic OT: $P^* = \\arg\\min_P \\langle P, C \\rangle + \\lambda H(P)$
with $C_{ij} = \\lVert a_i - b_j \\rVert^2$ and marginals $p, q$ proportional
to cluster sizes.

When K=1 this reduces exactly to mean_diff (one cluster each, P trivial,
kernel constant, single translation $b - a$).

Apply: $h \\leftarrow h + \\alpha \\hat v(x)$ (additive form, Definition 3.1).
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
class CHaRSC(SteeringConfig):
    method: str = "chars"
    k: int = 4
    n_kmeans_iters: int = 20
    lambda_: float = 0.1  # Sinkhorn entropy regularization
    n_sinkhorn: int = 50
    sigma: float | None = None  # None -> median pairwise distance heuristic


def _kmeans_counts(X: Tensor, k: int, n_iters: int, seed: int) -> tuple[Tensor, Tensor]:
    """Cosine-assignment k-means centroids and final cluster masses."""
    if k > X.shape[0]:
        raise ValueError(f"k={k} exceeds n={X.shape[0]}")
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=g)[:k]].clone()
    assign = torch.zeros(X.shape[0], dtype=torch.long)
    for _ in range(n_iters):
        Xn = X / (X.norm(dim=1, keepdim=True) + ε)
        Cn = C / (C.norm(dim=1, keepdim=True) + ε)
        assign = einsum(Xn, Cn, "n d, k d -> n k").argmax(dim=1)
        counts = torch.bincount(assign, minlength=k)
        if (counts == 0).any():
            raise ValueError("CHaRS k-means produced an empty cluster")
        new_C = torch.stack([X[assign == j].mean(0) for j in range(k)])
        if torch.allclose(new_C, C, atol=1e-6):
            C = new_C
            break
        C = new_C
    counts = torch.bincount(assign, minlength=k).float()
    return C, counts / counts.sum()


def _sinkhorn(C: Tensor, p: Tensor, q: Tensor, lam: float, n_iters: int) -> Tensor:
    """Entropic OT: P* = argmin <P,C> + lam*H(P), s.t. P 1 = p, P^T 1 = q."""
    K = torch.exp(-C / lam)
    u = torch.ones_like(p)
    for _ in range(n_iters):
        v = q / (K.t() @ u)
        u = p / (K @ v)
    return u[:, None] * K * v[None, :]


@register
class CHaRS:
    name = "chars"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: CHaRSC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            X_a = neg_acts[li].float()  # source = negative (steer from)
            X_b = pos_acts[li].float()  # target = positive (steer toward)
            if cfg.k > min(X_a.shape[0], X_b.shape[0]):
                raise ValueError(f"k={cfg.k} exceeds source/target sample count")

            a, p_marg = _kmeans_counts(X_a, k=cfg.k, n_iters=cfg.n_kmeans_iters, seed=cfg.seed)
            b, q_marg = _kmeans_counts(X_b, k=cfg.k, n_iters=cfg.n_kmeans_iters, seed=cfg.seed + 1)

            C = torch.cdist(a, b) ** 2
            P = _sinkhorn(C, p_marg, q_marg, cfg.lambda_, cfg.n_sinkhorn)

            # repo-stable global bandwidth heuristic; paper discusses an x-local median.
            if cfg.sigma is None:
                d_mat = torch.cdist(a, a)
                sigma = 1.0 if cfg.k == 1 else float(d_mat[d_mat > 0].median())
            else:
                sigma = float(cfg.sigma)

            out[li] = {"a": a, "b": b, "P": P, "p": p_marg, "sigma": torch.tensor(sigma)}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: CHaRSC,
    ) -> Float[Tensor, "b s d"]:
        a = state["a"].to(h)
        b = state["b"].to(h)
        P = state["P"].to(h)
        p = state["p"].to(h)
        sigma = float(state["sigma"])

        # k(x, a_i) per token. cdist doesn't support bf16 -- cast just for the
        # distance, then drop back to h's dtype for the rest of the math.
        d2    = (torch.cdist(h.float(), a.float()) ** 2).to(h.dtype)
        kern  = torch.exp(-d2 / (2 * sigma ** 2))
        denom = einsum(kern, p, "b s i, i -> b s")

        # weights w_ij(x) = P_ij * k_i(x) / sum_p p_p k_p(x)
        # v_hat(x) = sum_ij w_ij (b_j - a_i)
        w     = einsum(P, kern / denom[..., None], "i j, b s i -> b s i j")
        v_hat = einsum(w, b, "b s i j, j d -> b s d") - einsum(w, a, "b s i j, i d -> b s d")

        return h + cfg.coeff * v_hat
