"""Gated subspace ablation -- composes sspace + directional_ablation + cosine_gated.

Three operations stacked on the same SVD subspace:

1. sspace identifies the steering subspace $V_r \\in \\mathbb R^{d\\times r}$ via
   SVD of paired diffs and produces a normalized direction $\\hat v \\in
   \\mathrm{span}(V_r)$ as the projection of $\\bar D$ onto $V_r$.
2. **Subspace ablation** projects the entire $r$-dim subspace $V_r$ out of the
   residual stream (not just along $\\hat v$). This is what differentiates the
   method from plain directional_ablation.
3. **Cosine gate** scales the post-ablation nudge along $\\hat v$ by the
   $V_r$-space cosine of $h$ with $\\hat v$.

$$\\bar D = \\mathrm{mean}(H^+ - H^-),\\quad U S V^T = \\mathrm{SVD}(H^+ - H^-)$$
$$\\hat v = V_r V_r^T \\bar D \\ /\\ \\|V_r V_r^T \\bar D\\|$$
$$g = \\mathrm{relu}(|\\cos(V_r^T h,\\, V_r^T \\hat v)| - \\tau)$$
$$h \\leftarrow h - V_r V_r^T h + g \\cdot \\alpha \\cdot \\hat v$$

Why these compose:
- Subspace ablation $h - V_r V_r^T h$ removes the entire $r$-dim authority
  subspace, not just the 1-D direction $\\hat v$. Plain directional_ablation only
  removes the 1-D component along $\\hat v$ and is recovered when $r=1$.
- Cosine is computed in $V_r$-space ($V_r^T h$ vs $V_r^T \\hat v$). Since
  $\\hat v \\in \\mathrm{span}(V_r)$, $V_r^T \\hat v$ has unit norm and
  $\\cos(V_r^T h, V_r^T \\hat v) = \\langle h, \\hat v \\rangle / \\|V_r^T h\\|$.
  This differs from the full-d cosine because the denominator uses $\\|V_r^T h\\|$
  (subspace energy), not $\\|h\\|$ -- so the gate fires on tokens whose
  *subspace* component aligns with $\\hat v$, ignoring out-of-subspace mass.
- Gate is computed pre-ablation: post-ablation $h$ has zero $V_r$-component by
  construction, so the gate would be 0. Pre-ablation gate captures "how much
  $V_r$-signal was there" and scales the replacement nudge to match.

Hyperparameters $r$, $\\tau$, $\\alpha$ are not strictly orthogonal: changing $r$
rotates $\\hat v$ within the subspace and re-shapes the cosine distribution, so
$\\alpha$ must be re-calibrated (iso-KL). Default $\\tau=0$ makes the gate a
soft proportional dial in $[0, 1]$.
"""
from dataclasses import dataclass
import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class GatedSubspaceAblationC(SteeringConfig):
    method: str = "gated_subspace_ablation"
    r: int = 4
    tau: float = 0.0


@register
class GatedSubspaceAblation:
    name = "gated_subspace_ablation"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "n d"]],
        cfg: GatedSubspaceAblationC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            if pos_acts[li].shape[0] != neg_acts[li].shape[0]:
                raise ValueError(f"layer {li}: pos/neg counts differ")
            if cfg.r > min(pos_acts[li].shape[0], pos_acts[li].shape[1]):
                raise ValueError(f"r={cfg.r} exceeds rank bound")

            diffs = (pos_acts[li] - neg_acts[li]).float()
            mu    = diffs.mean(0)

            U, S, Vh = torch.linalg.svd(diffs, full_matrices=False)
            V = Vh[:cfg.r].T.contiguous()
            S = S[:cfg.r].contiguous()

            v = V @ (V.T @ mu)
            v = v / (v.norm() + ε)

            out[li] = {"v": v, "V": V, "S": S}
        return out

    @staticmethod
    def apply(
        block,
        h: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: GatedSubspaceAblationC,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(h)  # unit, in span(V)
        V = state["V"].to(h)  # (d, r), orthonormal columns from SVD
        S = state["S"].to(h)  # (r,), singular values

        # V-space coords: project into subspace.
        h_V = einsum(h, V, "b s d, d r -> b s r")      # V^T h
        v_V = V.T @ v                                   # V^T v_hat

        # S-space coords: weight by singular values so dominant
        # directions count more in the alignment measure.
        hS = h_V * S                                    # (b, s, r)
        vS = v_V * S                                    # (r,)

        # Cosine similarity in S-space.
        cos = ((hS * vS).sum(dim=-1, keepdim=True)
               / (hS.norm(dim=-1, keepdim=True) + ε)
               / (vS.norm() + ε))
        gate = torch.relu(cos.abs() - cfg.tau)

        # Subspace ablation: scaled by |coeff| so c=0 -> identity.
        h_proj = einsum(h_V, V, "b s r, d r -> b s d")
        h = h - abs(cfg.coeff) * h_proj

        # Replacement nudge along v_hat, gated by S-space alignment.
        return h + gate * cfg.coeff * v
