"""Method protocol + registry.

Each steering method implements two functions:

- `extract(pos_acts, neg_acts, cfg) -> dict[layer_idx, dict[str, Tensor]]`
    Given last-token hidden states for positive and negative prompts at each
    selected layer, compute per-layer state (vector, basis, centroids, ...) to
    be stored as buffers and read at runtime by `apply`.

- `apply(block, hidden_states, state, cfg) -> hidden_states`
    Called inside the forward hook on the block's output. `hidden_states` is
    `[batch, seq, d_model]`. `state` is the dict produced by `extract` for
    this block's layer index, as registered buffers on the block module.

State tensors are registered as buffers on the block module under
`_steering_state_<key>` (e.g. `_steering_state_v` for a single direction). The
hook reads them and rebuilds the state dict.
"""
from typing import Protocol, Any
import torch
from torch import Tensor


class Method(Protocol):
    name: str

    @staticmethod
    def extract(
        pos_acts: dict[int, Tensor],
        neg_acts: dict[int, Tensor],
        cfg: Any,
    ) -> dict[int, dict[str, Tensor]]:
        """Per-layer state. `pos_acts[l]` is `[n_pos, d_model]`, same for neg."""
        ...

    @staticmethod
    def apply(
        block,
        hidden_states: Tensor,  # [b, s, d]
        state: dict[str, Tensor],
        cfg: Any,
    ) -> Tensor:
        """Return modified hidden_states."""
        ...


REGISTRY: dict[str, type] = {}


def register(cls):
    if not getattr(cls, "name", None):
        raise ValueError(f"method {cls} missing .name")
    REGISTRY[cls.name] = cls
    return cls
