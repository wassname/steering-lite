r"""PCA in CorDA's context-oriented decomposition space.

CorDA (Yang et al. 2024) orients each Linear's weight decomposition with a
context covariance. For a Linear weight `W` and input-activation covariance
`Sigma_x`, CorDA decomposes:

$$W \Sigma_x = U S V^T$$

and reconstructs the original weight by applying the inverse covariance to the
right singular factor:

$$W = U S (\Sigma_x^{-1} V)^T$$

The reference implementation uses full-sequence covariance matrices and notes
large memory use for 7B-scale models. This steering variant uses the same
decomposition algebra on the last-token prompt activations used by the rest of
steering-lite, with Tikhonov damping:

$$\Sigma_\lambda = X^T X / n + \lambda I$$

Then it runs PCA on paired differences in the adapter hidden coordinate:

$$z = x (\Sigma_\lambda^{-1} V_r) \sqrt{S_r}$$

and applies a constant hidden-coordinate nudge:

$$y \leftarrow y + \alpha (v_z \sqrt{S_r}) U_r^T$$

Refs:
  - CorDA paper: https://arxiv.org/abs/2406.05223
  - Reference code: https://github.com/iboing/CorDA
"""
from dataclasses import dataclass

import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register
from .sspace_pca import _pca_direction


eps = 1e-8


@register_config
@dataclass
class CordaPCAC(SteeringConfig):
    method: str = "corda_pca"
    r: int = -1
    damping: float = 0.01
    normalize: bool = True


def _regularized_cov_inv_mul(X: Tensor, Y: Tensor, lam: Tensor) -> Tensor:
    """Return `(X.T X / n + lam I)^-1 Y` without forming the d x d inverse."""
    n = X.shape[0]
    B = X / n**0.5
    gram = torch.eye(n, dtype=X.dtype, device=X.device) + (B @ B.T) / lam
    return Y / lam - B.T @ torch.linalg.solve(gram, B @ Y) / (lam * lam)


def _corda_basis(W: Tensor, X: Tensor, r: int, damping: float) -> tuple[Tensor, Tensor, Tensor]:
    W = W.float()
    X = X.to(device=W.device, dtype=torch.float32)
    n = X.shape[0]
    lam = X.square().mean() * damping
    if lam <= 0:
        raise ValueError(f"non-positive CorDA damping lambda {lam.item():.3e}")

    # W Sigma_lambda = W (X.T X / n + lambda I), without materializing Sigma.
    W_sigma = (W @ X.T) @ X / n + lam * W
    U, S, Vh = torch.linalg.svd(W_sigma, full_matrices=False)
    k = min(W.shape)
    r_eff = k if r < 0 or r >= k else r
    U_r = U[:, :r_eff].cpu().contiguous()
    sqrtS = S[:r_eff].sqrt().cpu().contiguous()
    V_raw = Vh[:r_eff].T.contiguous()
    V_corda = _regularized_cov_inv_mul(X, V_raw, lam).cpu().contiguous()
    return U_r, sqrtS, V_corda


@register
class CordaPCA:
    name = "corda_pca"
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"
    record_linear_inputs = True

    @staticmethod
    def extract(
        pos_inputs: dict[str, Float[Tensor, "n d_in"]],
        neg_inputs: dict[str, Float[Tensor, "n d_in"]],
        cfg: CordaPCAC,
        *,
        name_to_module: dict[str, torch.nn.Module],
    ) -> dict[str, dict[str, dict[str, Tensor]]]:
        out = {}
        for name, x_pos in pos_inputs.items():
            mod = name_to_module[name]
            X = torch.cat([x_pos.float(), neg_inputs[name].float()], dim=0)
            U_r, sqrtS, V_corda = _corda_basis(mod.weight, X, cfg.r, cfg.damping)

            z_pos = (x_pos.float() @ V_corda) * sqrtS
            z_neg = (neg_inputs[name].float() @ V_corda) * sqrtS
            v = _pca_direction(z_pos - z_neg, cfg.normalize)

            out[name] = {
                "shared": {"U_r": U_r, "sqrtS": sqrtS, "V_corda": V_corda},
                "stacked": {"v": v.unsqueeze(0)},
            }
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d_in"],
        y: Float[Tensor, "b s d_out"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: CordaPCAC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r = shared["U_r"].to(y)
        sqrtS = shared["sqrtS"].to(y)
        v = stacked["v"].to(y).sum(dim=0)
        return y + cfg.coeff * (v * sqrtS) @ U_r.T
