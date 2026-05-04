"""Weight-SVD S-space steering, cosine-gated (AntiPaSTO arithmetic relaxation).

Standard activation steering adds a constant bias `h <- h + alpha * v`
regardless of input. Weight-SVD S-space steering operates in the SVD basis of
a Linear's *weight matrix*, so the perturbation is input-dependent: tokens
whose S-space representation aligns with the contrastive direction get
pushed; tokens that don't are left alone.

For a Linear `y = x W^T + b` with `W = U S V^T` truncated to top-r, the
whitened S-space coordinates are accessible from EITHER side:

    x V sqrt(S) = (y - b) U / sqrt(S) = x_S          # x_S == y_S

We use the *output* projection at both extract and apply time:

    x_S = (y - b) @ U / sqrt(S)        # [..., r]

Output capture is cheaper than input capture for residual-writer Linears
(e.g. `mlp.down_proj`: d_out=2560 vs d_in=9728 on Qwen3.5-4B). Residual-
reader Linears (q/k/v/up/gate proj) would prefer the input side.

Default target: residual writers (`mlp.down_proj` AND `self_attn.o_proj`).
Override via `cfg.target_submodule = <regex>` -- e.g.
`r"self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj"` for all 7 Linears.

Contrastive direction in S-space (from POS/NEG persona-branching pairs):

    d_S     = mean(x_S^+) - mean(x_S^-)
    d_S_hat = d_S / ||d_S||

Cosine gate (sign-agnostic, matches `cosine_gated.py`):

    gate     = |cos(x_S, d_S_hat)|              in [0, 1]
    delta_S  = alpha * gate * d_S_hat           [..., r]
    delta_y  = (delta_S * sqrt(S)) @ U_r^T      lift S-space delta back to out-space
    y'       = y + delta_y

This is the arithmetic relaxation of AntiPaSTO (Clark, 2026): same S-space
parameterization, but contrastive mean-diff extraction instead of gradient
optimization.

Why cosine in S-space (not residual d-space): in d=2560 the cosine of two
random vectors concentrates near 0 -- the gate degenerates. In r=64 the
cosine is a meaningful per-token signal; tokens activating dominant S-space
modes get a strong gate, tokens in the tail get near-zero.
"""
from dataclasses import dataclass
import torch
from jaxtyping import Float
from torch import Tensor

from ..config import SteeringConfig, register_config, register


ε = 1e-8


@register_config
@dataclass
class SSpaceC(SteeringConfig):
    method: str = "sspace"
    r: int = -1  # -1 = full rank (no cropping); else top-r modes by |d_S|


def _orient_svd(
    U: Float[Tensor, "d_out r"],
    S: Float[Tensor, "r"],
    Vh: Float[Tensor, "r d_in"],
    h_interleaved: Float[Tensor, "n2 d_out"],
) -> tuple[Tensor, Tensor, Tensor]:
    """Orient SVD sign so +coeff = toward positive persona (repeng majority-vote).

    SVD signs per component are arbitrary. Project samples through U, count
    how often pos (even idx) > neg (odd idx); flip components where pos<neg.
    Caller passes pos/neg interleaved (same n on each side).
    """
    proj: Float[Tensor, "n r"] = h_interleaved @ U
    positive_larger: Float[Tensor, "r"] = (proj[::2] > proj[1::2]).to(U.dtype).mean(0)
    flip: Float[Tensor, "r"] = torch.where(
        positive_larger < 0.5,
        torch.tensor(-1.0),
        torch.tensor(1.0),
    )
    return U * flip, S, Vh * flip[:, None]


@register
class SSpace:
    name = "sspace"
    # Default: hook both residual writers per block. Multi-submodule via regex.
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"

    @staticmethod
    def extract(
        pos_outputs: dict[str, Float[Tensor, "n d_out"]],
        neg_outputs: dict[str, Float[Tensor, "n d_out"]],
        cfg: SSpaceC,
        *,
        name_to_module: dict[str, torch.nn.Module],
    ) -> dict[str, dict[str, Tensor]]:
        out = {}
        for name, y_pos in pos_outputs.items():
            mod = name_to_module[name]
            W = mod.weight  # [d_out, d_in]
            k = min(W.shape)
            r_eff = k if (cfg.r is None or cfg.r < 0 or cfg.r >= k) else cfg.r
            U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
            U, S, Vh = U.cpu(), S.cpu(), Vh.cpu()

            b = mod.bias.detach().cpu().float() if mod.bias is not None else None
            y_pos_f = y_pos.float()
            y_neg_f = neg_outputs[name].float()
            if b is not None:
                y_pos_f = y_pos_f - b
                y_neg_f = y_neg_f - b

            # repeng-style sign flip: pos/neg interleaved so proj[::2] vs proj[1::2] is pos vs neg
            n = min(y_pos_f.shape[0], y_neg_f.shape[0])
            h_il = torch.stack([y_pos_f[:n], y_neg_f[:n]], dim=1).flatten(0, 1)  # [2n, d_out]
            U, S, Vh = _orient_svd(U, S, Vh, h_il)

            # full S-space projection (V implicit via identity (y-b)U/sqrt(S) = xV sqrt(S))
            sqrtS_full = S.sqrt()
            xS_pos = (y_pos_f @ U) / sqrtS_full              # [n, k]
            xS_neg = (y_neg_f @ U) / sqrtS_full              # [n, k]
            dS = xS_pos.mean(0) - xS_neg.mean(0)             # [k]

            # task-specific mode selection: top-r modes by contrastive magnitude.
            # r_eff == k means full rank (no cropping).
            if r_eff < k:
                mask = dS.abs().topk(r_eff).indices.sort().values
                U_r = U[:, mask].contiguous()                # [d_out, r]
                Vh_r = Vh[mask, :].contiguous()              # [r, d_in]
                sqrtS = sqrtS_full[mask].contiguous()        # [r]
                dS_r = dS[mask]
            else:
                U_r, Vh_r = U.contiguous(), Vh.contiguous()
                sqrtS = sqrtS_full
                dS_r = dS
            dS_hat = (dS_r / (dS_r.norm() + ε)).contiguous()  # [r]

            entry = {"U_r": U_r, "sqrtS": sqrtS, "dS_hat": dS_hat, "Vh_r": Vh_r}
            if b is not None:
                entry["b"] = b
            out[name] = entry
        return out

    @staticmethod
    def apply(
        block,
        x: Float[Tensor, "b s d_in"],     # unused; kept for hook signature
        y: Float[Tensor, "b s d_out"],
        state: dict[str, Tensor],
        cfg: SSpaceC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = state["U_r"].to(y)
        sqrtS  = state["sqrtS"].to(y)
        dS_hat = state["dS_hat"].to(y)
        y_eff = y - state["b"].to(y) if "b" in state else y

        xS = (y_eff @ U_r) / sqrtS                                  # [b, s, r]
        xS_norm = xS / (xS.norm(dim=-1, keepdim=True) + ε)
        cos = (xS_norm * dS_hat).sum(dim=-1, keepdim=True)          # [b, s, 1]
        gate = cos.abs()                                             # [0, 1]

        delta_S = cfg.coeff * gate * dS_hat                         # [b, s, r]
        delta_y = (delta_S * sqrtS) @ U_r.T                         # [b, s, d_out]
        return y + delta_y
