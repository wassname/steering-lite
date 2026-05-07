"""Super-SVD S-space steering on the residual stream (cosine-gated, multi-vec).

Like sspace, but the basis is shared across many Linears. Where sspace SVDs
ONE weight matrix and steers in its column-space, super_sspace pools the
residual-side singular vectors of ALL writers and readers in the selected
blocks and SVDs the pool. The result is a global d_model -> r basis that
covers what the residual stream can hold from layer activity, not just one
Linear's slice.

Math (writers W with d_out=d_model, readers W with d_in=d_model):

    For non-square W = U Σ V^T:
      writer block:  B_l = U_l Σ_l        ∈ R^{d_model × k_l}
      reader block:  B_l = V_l Σ_l        ∈ R^{d_model × k_l}

    Direct: M = [B_1 | B_2 | ...] ∈ R^{d_model × Σ k_l}, then SVD M -> U_⋆.
    Cheaper: SVD via the Gram matrix:

      G = M M^T = Σ_l B_l B_l^T = Σ_l U_l Σ_l² U_l^T  (and V_l Σ_l² V_l^T)
                                    ∈ R^{d_model × d_model}

      eig(G) -> λ_⋆, U_⋆;  Σ_⋆ = sqrt(λ_⋆)

    G is d_model × d_model (e.g. 2560²) regardless of how many Linears we
    pool. Same U_⋆, Σ_⋆ as direct SVD of M.

Per-layer dS (residual-stream activations at hook layer L, n examples each):

    xS_pos = h_pos[L] @ U_⋆ / sqrt(Σ_⋆)
    xS_neg = h_neg[L] @ U_⋆ / sqrt(Σ_⋆)
    dS     = mean(xS_pos) - mean(xS_neg)
    select top-r modes by |dS| (task-specific, same as sspace).

Apply (block residual hook, like mean_diff):

    xS      = h @ U_⋆_r / sqrt(Σ_⋆_r)
    gate    = |cos(xS, dS_hat)|
    delta_S = α * gate * dS_hat
    delta_h = (delta_S * sqrt(Σ_⋆_r)) @ U_⋆_r^T
    h'      = h + delta_h

Compare to sspace: per-Linear basis vs global pooled basis. Hypothesis: the
pooled basis is more robust because tail directions of any single Linear are
noisy, but the *consensus* of writers/readers is cleaner.

Multi-round: basis is purely W-derived (Gram eigh of
writers/readers), so the shared-basis invariant holds across iterated rounds.
dS lives in `stacked` with leading [k, r] dim; apply mirrors sspace's einsum
pattern -- per-direction cosine gate, per-direction calibration via row norm.

Square Linears (e.g. Qwen3 o_proj / q_proj where d_in=d_out=d_model) are
ambiguous (writer or reader?) and are skipped by shape detection. Pass a
custom `fallback_regex` if your arch has unusual shapes.
"""
from dataclasses import dataclass
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register
from ..target import find_residual_linears


ε = 1e-8


@register_config
@dataclass
class SuperSSpaceC(SteeringConfig):
    method: str = "super_sspace"
    r: int = -1                  # -1 = full d_model rank; else top-r by |dS|
    role: str = "both"           # "writer" | "reader" | "both"


@register
class SuperSSpace:
    name = "super_sspace"
    needs_model = True           # extract receives `model=` kwarg from train()

    @staticmethod
    def _build_super_basis(model, layer_indices, role):
        """Pool U Σ (writers) and V Σ (readers) into Gram, eigendecompose."""
        from loguru import logger
        targets = find_residual_linears(model, layer_indices, role=role)
        if not targets:
            raise RuntimeError("no residual writers/readers found")

        d_model = next(iter(t[1].weight.shape[0] for t in targets if t[3] == "writer")
                       if any(t[3] == "writer" for t in targets)
                       else iter(t[1].weight.shape[1] for t in targets))

        G = torch.zeros(d_model, d_model, dtype=torch.float64)
        n_w = n_r = 0
        for _, mod, _, side in targets:
            W = mod.weight.detach().cpu().double()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            if side == "writer":
                # B = U Σ, so B B^T = U Σ² U^T
                G += U @ torch.diag(S * S) @ U.T
                n_w += 1
            else:  # reader: residual-side basis is V (rows of Vh)
                V = Vh.T
                G += V @ torch.diag(S * S) @ V.T
                n_r += 1
        # eigh returns ascending eigenvalues; reverse for descending σ²
        λ, U_super = torch.linalg.eigh(G)
        λ = λ.flip(0).clamp_min(0)
        U_super = U_super.flip(1)
        Σ_super = λ.sqrt()                                # [d_model]
        logger.info(f"super_sspace basis: d_model={d_model}, "
                    f"writers={n_w}, readers={n_r}, "
                    f"σ_⋆[0]={Σ_super[0].item():.3g} σ_⋆[-1]={Σ_super[-1].item():.3g}")
        return U_super.float(), Σ_super.float()

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d_model"]],
        neg_acts: dict[int, Float[Tensor, "n d_model"]],
        cfg: SuperSSpaceC,
        *,
        model,
    ) -> dict[int, dict[str, Tensor]]:
        layer_indices = tuple(pos_acts.keys())
        U_super, Σ_super = SuperSSpace._build_super_basis(model, layer_indices, cfg.role)
        sqrtΣ = Σ_super.sqrt().clamp_min(ε)               # [d_model]; clamp avoids /0 for zero modes

        d_model = U_super.shape[0]
        r_eff = d_model if (cfg.r is None or cfg.r < 0 or cfg.r >= d_model) else cfg.r

        out = {}
        for li, h_pos in pos_acts.items():
            h_pos_f = h_pos.float()
            h_neg_f = neg_acts[li].float()
            xS_pos = (h_pos_f @ U_super) / sqrtΣ          # [n, d_model]
            xS_neg = (h_neg_f @ U_super) / sqrtΣ
            dS = xS_pos.mean(0) - xS_neg.mean(0)          # [d_model]

            if r_eff < d_model:
                idx = dS.abs().topk(r_eff).indices.sort().values
                U_r = U_super[:, idx].contiguous()
                sqrtS_r = sqrtΣ[idx].contiguous()
                dS_r = dS[idx]
            else:
                U_r, sqrtS_r, dS_r = U_super.contiguous(), sqrtΣ.contiguous(), dS

            # Normalize by the residual-space magnitude of dS: ||(dS_r * sqrtS_r)||.
            # S-space normalization (||dS_r||=1) is wrong for super_sspace because
            # Gram eigenvalues (sqrtS) are pooled across ALL Linears and can be
            # orders of magnitude larger than any single Linear's σ. The roundtrip
            # (dS_unit * sqrtS) @ U_r.T would then produce a huge residual delta
            # at any nonzero gate, causing NaN at k=2. Residual-space normalization
            # makes alpha=||stacked[dS] row|| directly equal to the residual delta
            # magnitude, matching mean_diff's coeff semantics.
            dS_residual_norm = ((dS_r * sqrtS_r).norm() + ε)
            dS_unit = (dS_r / dS_residual_norm).contiguous()
            out[li] = {
                "shared": {"U_r": U_r, "sqrtS": sqrtS_r},
                "stacked": {"dS": dS_unit.unsqueeze(0)},   # [1, r] on first round
            }
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: SuperSSpaceC,
    ) -> Float[Tensor, "b s d"]:
        U_r    = shared["U_r"].to(y)
        sqrtS  = shared["sqrtS"].to(y)

        xS = (y @ U_r) / sqrtS                                            # [b, s, r]
        xS_norm = xS / (xS.norm(dim=-1, keepdim=True) + ε)

        dS_raw = stacked["dS"].to(y)                                       # [k, r]
        alpha  = dS_raw.norm(dim=-1)                                       # [k]
        dS_hat = dS_raw / (alpha.unsqueeze(-1) + ε)                        # [k, r] unit

        gate    = einsum(xS_norm, dS_hat, "b s r, k r -> b s k").abs()    # [b, s, k]
        weighted = gate * alpha                                            # [b, s, k]
        delta_S = einsum(weighted, dS_hat, "b s k, k r -> b s r")         # [b, s, r]
        return y + cfg.coeff * (delta_S * sqrtS) @ U_r.T
