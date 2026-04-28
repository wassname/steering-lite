"""Record last non-pad-token hidden states at selected layers via forward hooks.

We hook each block's forward output (it returns `(hidden_states, ...)`), gather
the final non-padding token from `attention_mask`, and stack across prompts.
No grad.
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
    captured: dict[int, Tensor] = {}

    def make_hook(li: int):
        def hook(_mod, _args, out):
            h = out[0] if isinstance(out, tuple) else out
            captured[li] = h
        return hook

    handles = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]
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
            for li in layers:
                bucket[li].append(captured[li][batch_idx, last_idx].detach().to("cpu"))
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {li: torch.cat(bucket[li], dim=0) for li in layers}
