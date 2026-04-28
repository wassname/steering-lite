"""Find transformer blocks to hook.

We hook the **block** module (each entry in `model.model.layers`) on its forward
output. The block returns `(hidden_states, ...)`; we add the steering update to
`hidden_states` and return the modified tuple.

Works with HF llama-family architectures (llama, qwen, mistral, etc). For other
architectures, set `cfg.layers` to indices into whatever list lives at the path
your model exposes -- override `_get_blocks` if needed.
"""
from torch import nn


def _get_blocks(model: nn.Module) -> nn.ModuleList:
    # llama-family: model.model.layers
    inner = getattr(model, "model", model)
    blocks = getattr(inner, "layers", None)
    if blocks is None:
        raise RuntimeError(
            f"could not find .model.layers on {type(model).__name__}; "
            f"override _get_blocks for non-llama architectures"
        )
    return blocks


def find_targets(model: nn.Module, cfg) -> list[tuple[str, nn.Module, int]]:
    """Return [(name, block_module, layer_idx)] for layers selected by cfg."""
    blocks = _get_blocks(model)
    n = len(blocks)
    if cfg.layers is None:
        idxs = tuple(range(n))
    else:
        idxs = tuple(cfg.layers)
        for i in idxs:
            if not (0 <= i < n):
                raise ValueError(f"layer {i} out of range [0, {n})")
    return [(f"layers.{i}", blocks[i], i) for i in idxs]


def get_d_model(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    d = getattr(cfg, "hidden_size", None)
    if d is None:
        raise RuntimeError("model has no .config.hidden_size")
    return int(d)
