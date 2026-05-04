"""Record last non-pad-token *outputs* of named sub-modules via forward hooks.

Companion to `extract.record_activations`, but for variants that need a
specific Linear's output (e.g. weight-SVD steering). For a Linear with
`W = U S V^T`, the (whitened) S-space input coordinates `xS = x V sqrt(S)`
can be recovered from the output without ever seeing x:

    (y - b) = x V S U^T
    (y - b) @ U / sqrt(S) = x V sqrt(S) = xS

Keyed by full module name (str) so multiple submodules per block work
naturally (e.g. residual writers mlp.down_proj + self_attn.o_proj together).

Returns `dict[full_name, Tensor[n, d_out]]`.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from jaxtyping import Float


@torch.no_grad()
def record_linear_outputs(
    model: nn.Module,
    tok,
    prompts: list[str],
    name_to_module: dict[str, nn.Module],
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[str, Float[Tensor, "n d_out"]]:
    """Forward-hook each named sub-module; record last non-pad output row per prompt."""
    device = next(model.parameters()).device

    bucket: dict[str, list[Tensor]] = {name: [] for name in name_to_module}
    captured: dict[str, Tensor] = {}

    def make_hook(name: str):
        def hook(_mod, _args, out):
            captured[name] = out
        return hook

    handles = [m.register_forward_hook(make_hook(name))
               for name, m in name_to_module.items()]
    try:
        was_training = model.training
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            captured.clear()
            model(**enc)
            mask = enc["attention_mask"]
            last_idx = mask.shape[1] - 1 - mask.flip([-1]).argmax(-1)
            batch_idx = torch.arange(mask.shape[0], device=last_idx.device)
            for name in name_to_module:
                bucket[name].append(captured[name][batch_idx, last_idx].detach().to("cpu"))
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {name: torch.cat(bucket[name], dim=0) for name in name_to_module}
