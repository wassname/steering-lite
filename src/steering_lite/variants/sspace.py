"""Weight-SVD S-space steering, cosine-gated (AntiPaSTO arithmetic relaxation).

Standard activation steering adds a constant bias `h <- h + alpha * v`
regardless of input. Weight-SVD S-space steering operates in the SVD basis of
a Linear's *weight matrix*, so the perturbation is input-dependent: tokens
whose S-space representation aligns with the contrastive direction get
pushed; tokens that don't are left alone.

For a Linear `y = x W^T + b` with `W = U S V^T` truncated to top-r, the
whitened S-space coordinates are accessible from EITHER side:

    x V sqrt(S) = (y - b) U / sqrt(S) = x_S          # x_S == y_S

We use the *output* projection at both extract and apply time.

Multi-round:

  - SVD basis (U_r, sqrtS, Vh_r, b) is a property of the weight matrix and
    is `shared` across rounds. Each round's `dS_2`, `dS_3`, ... extracts in
    the SAME basis (since W is frozen), which is exactly what makes
    accumulation valid.
  - Stacked tensor `dS: [k, r]`. Each row is `alpha_i * dS_hat_i_unit`,
    so row magnitude carries that round's per-direction calibration. Apply
    normalizes on-the-fly.
  - Per-round gate: each direction keeps its own `|cos(xS, dS_i)|`. This is
    strictly more faithful than baking magnitudes into a single direction,
    because each contrast was calibrated under different conditions.

Apply (k stacked directions, all in the same basis):

    xS      = (y - b) @ U_r / sqrt(S)              # [b, s, r]   ONCE
    dS_hat  = dS / ||dS||_row                       # [k, r] unit
    alpha   = ||dS||_row                            # [k]   per-direction calib
    gate    = |cos(xS, dS_hat)|                     # [b, s, k]
    deltaS  = einsum(gate * alpha, dS_hat, "bsk,kr->bsr") * cfg.coeff
    y'      = y + (deltaS * sqrt(S)) @ U_r^T

Why cosine in S-space: in d=2560 the cosine of two random vectors
concentrates near 0; in r=64 the cosine is a meaningful per-token signal.
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
class SSpaceC(SteeringConfig):
    method: str = "sspace"
    r: int = -1  # -1 = full rank (no cropping); else top-r modes by |d_S|
    gate: str = "cosine"  # "cosine" = |cos(xS, dS_hat)| per-token; "off" = constant gate=1
                          # (sums stacked dS rows; behaves like mean_diff in S-basis)


# Removed _orient_svd: previously flipped SVD column signs based on
# activation-contrast majority vote so +coeff aligned with the positive persona.
# That orientation depends on the contrast, which DIFFERS in iterated rounds
# (round 2's extract runs under v_running attached), so U_r flipped between
# rounds and `Vector + Vector`'s shared-basis invariant broke.
# Now the basis is purely W-derived (torch.linalg.svd is deterministic).
# Sign of dS in the resulting basis is arbitrary; iterated_steer.py already
# picks ±C at apply time, and single-round users can pass -coeff if needed.


@register
class SSpace:
    name = "sspace"
    default_target_submodule = r"mlp\.down_proj|self_attn\.o_proj"

    @staticmethod
    def extract(
        pos_outputs: dict[str, Float[Tensor, "n d_out"]],
        neg_outputs: dict[str, Float[Tensor, "n d_out"]],
        cfg: SSpaceC,
        *,
        name_to_module: dict[str, torch.nn.Module],
    ) -> dict[str, dict[str, dict[str, Tensor]]]:
        out = {}
        for name, y_pos in pos_outputs.items():
            mod = name_to_module[name]
            W = mod.weight
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

            sqrtS_full = S.sqrt()
            xS_pos = (y_pos_f @ U) / sqrtS_full
            xS_neg = (y_neg_f @ U) / sqrtS_full
            dS = xS_pos.mean(0) - xS_neg.mean(0)

            if r_eff < k:
                mask = dS.abs().topk(r_eff).indices.sort().values
                U_r = U[:, mask].contiguous()
                Vh_r = Vh[mask, :].contiguous()
                sqrtS = sqrtS_full[mask].contiguous()
                dS_r = dS[mask]
            else:
                U_r, Vh_r = U.contiguous(), Vh.contiguous()
                sqrtS = sqrtS_full
                dS_r = dS
            dS_unit = (dS_r / (dS_r.norm() + ε)).contiguous()

            shared = {"U_r": U_r, "sqrtS": sqrtS, "Vh_r": Vh_r}
            if b is not None:
                shared["b"] = b
            stacked = {"dS": dS_unit.unsqueeze(0)}  # [1, r]; magnitude=1 on first round
            out[name] = {"shared": shared, "stacked": stacked}
        return out

    @staticmethod
    def apply(
        mod,
        x: Float[Tensor, "b s d_in"],
        y: Float[Tensor, "b s d_out"],
        shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: SSpaceC,
    ) -> Float[Tensor, "b s d_out"]:
        U_r    = shared["U_r"].to(y)
        sqrtS  = shared["sqrtS"].to(y)
        b      = shared.get("b")
        y_eff  = y - b.to(y) if b is not None else y

        dS_raw = stacked["dS"].to(y)                                       # [k, r]
        if cfg.gate == "off":
            deltaS = dS_raw.sum(dim=0)                                     # [r] constant per token
        elif cfg.gate == "cosine":
            xS = (y_eff @ U_r) / sqrtS                                    # [b, s, r]
            xS_norm = xS / (xS.norm(dim=-1, keepdim=True) + ε)
            alpha  = dS_raw.norm(dim=-1)                                   # [k]
            dS_hat = dS_raw / (alpha.unsqueeze(-1) + ε)                    # [k, r] unit
            gate    = einsum(xS_norm, dS_hat, "b s r, k r -> b s k").abs() # [b, s, k]
            weighted = gate * alpha                                        # [b, s, k]
            deltaS   = einsum(weighted, dS_hat, "b s k, k r -> b s r")    # [b, s, r]
        else:
            raise ValueError(f"unknown gate mode {cfg.gate!r}; expected 'cosine' or 'off'")
        return y + cfg.coeff * (deltaS * sqrtS) @ U_r.T
