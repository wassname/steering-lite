"""Mean-diff directional ablation (Arditi-inspired projection-out).

Project the steering direction *out of* the residual stream instead of (or in
addition to) adding to it. Unlike `mean_diff` which translates by $\\alpha v$,
ablation removes the component of $h$ along $\\hat v$:

$$h \\leftarrow h - (h \\cdot \\hat v)\\hat v + \\alpha\\hat v$$

When `coeff=0` this is pure ablation (refusal-direction style); when `coeff!=0`
this is ablation followed by a constant nudge (useful to ablate "old" behavior
and inject "new"). The two terms are mathematically distinct -- ablation is a
*projection* (idempotent), addition is a *translation*.

Norms shrink by $|h \\cdot \\hat v|$ which is informative -- a near-zero shrink
means the direction wasn't present in the first place, so the intervention is
a no-op. Compare to `mean_diff` which always pays a constant $\\alpha\\|\\hat v\\|$
per token regardless of whether the direction is present.

Refs / inspiration:
  - Arditi et al. 2024 "Refusal in language models is mediated by a single direction"
    https://arxiv.org/abs/2406.11717
  - andyrdt/refusal_direction https://github.com/andyrdt/refusal_direction
"""
from dataclasses import dataclass
from einops import einsum
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class DirectionalAblationC(SteeringConfig):
    method: str = "directional_ablation"
    coeff: float = 0.0  # post-ablation additive nudge along v_hat; 0.0 = pure ablation


@register
class DirectionalAblation:
    name = "directional_ablation"

    @staticmethod
    def extract(
        pos_acts: dict[int, Float[Tensor, "n d"]],
        neg_acts: dict[int, Float[Tensor, "m d"]],
        cfg: DirectionalAblationC,
    ) -> dict[int, dict[str, Tensor]]:
        out = {}
        for li in pos_acts:
            v = pos_acts[li].float().mean(0) - neg_acts[li].float().mean(0)
            v = v / (v.norm() + ε)

            out[li] = {"v": v}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        state: dict[str, Tensor],
        cfg: DirectionalAblationC,
    ) -> Float[Tensor, "b s d"]:
        v = state["v"].to(y)  # unit

        proj = einsum(y, v, "b s d, d -> b s")
        y    = y - proj.unsqueeze(-1) * v          # ablate

        if cfg.coeff != 0.0:
            y = y + cfg.coeff * v
        return y
