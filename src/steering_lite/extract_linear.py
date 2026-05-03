"""Record last non-pad-token *outputs* of a sub-module via forward hooks.

Companion to `extract.record_activations`, but for variants that need a
specific Linear's output (e.g. weight-SVD steering). For a Linear with
`W = U S V^T`, the (whitened) S-space input coordinates `xS = x V sqrt(S)`
can be recovered from the output without ever seeing x:

    (y - b) = x V S U^T
    (y - b) @ U / sqrt(S) = x V sqrt(S) = xS

so projecting y through `U / sqrt(S)` gives the same xS, and y is typically
smaller than x (e.g. mlp.down_proj has d_out < d_in). Variants that need xS
should record y here and project at extract time.

Returns `dict[layer_idx, Tensor[n, d_out]]` matching `record_activations`'s shape.
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
    layer_to_module: dict[int, nn.Module],
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[int, Float[Tensor, "n d_out"]]:
    """Forward-hook each sub-module; record last non-pad output row per prompt."""
    device = next(model.parameters()).device

    bucket: dict[int, list[Tensor]] = {li: [] for li in layer_to_module}
    captured: dict[int, Tensor] = {}

    def make_hook(li: int):
        def hook(_mod, _args, out):
            captured[li] = out
        return hook

    handles = [m.register_forward_hook(make_hook(li))
               for li, m in layer_to_module.items()]
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
            for li in layer_to_module:
                bucket[li].append(captured[li][batch_idx, last_idx].detach().to("cpu"))
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {li: torch.cat(bucket[li], dim=0) for li in layer_to_module}
