"""Record last-token hidden states at selected layers via forward hooks.

We hook each block's forward output (it returns `(hidden_states, ...)`), grab
`hidden_states[:, -1, :]`, and stack across prompts. No grad.
"""
from __future__ import annotations
import torch
from torch import nn, Tensor
from jaxtyping import Float

from .target import _get_blocks


@torch.no_grad()
def record_activations(
    model: nn.Module,
    tok,
    prompts: list[str],
    layers: tuple[int, ...],
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[int, Float[Tensor, "n d"]]:
    """Run prompts through model, return last-token hidden state at each layer."""
    blocks = _get_blocks(model)
    device = next(model.parameters()).device

    bucket: dict[int, list[Tensor]] = {l: [] for l in layers}

    def make_hook(li: int):
        def hook(_mod, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            bucket[li].append(h[:, -1, :].detach().to("cpu"))
        return hook

    handles = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        was_training = model.training
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
            model(**enc)
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {li: torch.cat(bucket[li], dim=0) for li in layers}
