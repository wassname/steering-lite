r"""PCA in whitened weight-SVD S-space.

This is the clean ablation between residual PCA and cosine-gated S-space:

1. For each target Linear, decompose `W = U S V^T`.
2. Recover whitened S-space coordinates from the Linear output:

$$x_S = (y - b) U_r / \sqrt{S_r}$$

3. Run PCA on paired differences `x_S^+ - x_S^-`.
4. At apply time, add the PCA direction in S-space and map it back:

$$y \leftarrow y + \alpha (v_S \sqrt{S_r}) U_r^T$$

Unlike `sspace`, there is no cosine gate. The only question is whether the
PCA direction estimator is better after moving from residual space into the
weight-SVD coordinates.
"""
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


eps = 1e-8


@register_config
@dataclass
class SSpacePCAC(SteeringConfig):
    method: str = "sspace_pca"
    r: int = -1
    normalize: bool = True


def _pca_direction(diffs: Tensor, normalize: bool) -> Tensor:
    centered = diffs - diffs.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    v = Vh[0]
    # (Claude 2026-07-15) Orient the sign-ambiguous top PC to the persona contrast
    # itself: sign(mean(diffs) . v), so +coeff always moves toward the positive pole.
    # The old vote was on CENTERED projections (mean-zero by construction), so it
    # measured the variance cloud's skew, not concept polarity, and flipped the
    # steering direction at random (sspace_pca landed on-axis<0, corda_pca on-axis>0
    # from this same fn). This is the AntiPaSTO/repeng "align to hs" rule:
    # AntiPaSTO_concepts/README.md:577-582 saliency = sign(mean(diff_S)) * std(diff_S).
    v = v * torch.sign(diffs.mean(0) @ v + eps)
    if normalize:
        v = v / (v.norm() + eps)
    return v.contiguous()


@register
class SSpacePCA:
    name = "sspace_pca"
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"

    @staticmethod
    def extract(
        pos_outputs: dict[str, Float[Tensor, "n d_out"]],
        neg_outputs: dict[str, Float[Tensor, "n d_out"]],
        cfg: SSpacePCAC,
        *,
        name_to_module: dict[str, torch.nn.Module],
    ) -> dict[str, dict[str, dict[str, Tensor]]]:
        out = {}
        for name, y_pos in pos_outputs.items():
            mod = name_to_module[name]
            W = mod.weight
            k = min(W.shape)
            r_eff = k if cfg.r < 0 or cfg.r >= k else cfg.r
            U, S, _Vh = torch.linalg.svd(W.float(), full_matrices=False)
            U_r = U[:, :r_eff].cpu().contiguous()
            sqrtS = S[:r_eff].sqrt().cpu().contiguous()

            b = mod.bias.detach().cpu().float() if mod.bias is not None else None
            y_pos_f = y_pos.float()
            y_neg_f = neg_outputs[name].float()
            if b is not None:
                y_pos_f = y_pos_f - b
                y_neg_f = y_neg_f - b

            z_pos = (y_pos_f @ U_r) / sqrtS
            z_neg = (y_neg_f @ U_r) / sqrtS
            v = _pca_direction(z_pos - z_neg, cfg.normalize)

            shared = {"U_r": U_r, "sqrtS": sqrtS}
            if b is not None:
                shared["b"] = b
            out[name] = {"shared": shared, "stacked": {"v": v.unsqueeze(0)}}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d_in"],
        y: Float[Tensor, "b s d_out"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: SSpacePCAC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r = shared["U_r"].to(y)
        sqrtS = shared["sqrtS"].to(y)
        v = stacked["v"].to(y).sum(dim=0)
        return y + cfg.coeff * (v * sqrtS) @ U_r.T
