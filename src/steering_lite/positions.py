"""Restrict steering to selected prompt tokens, e.g. the user's message. PI/OpenAI 2026-10-02.

    with vector(model, C=c), only_tokens(mask):  # mask [b, s] over the padded prompt
        model.generate(**batch)

During prefill (sequence length s) each steered output keeps its edit only where mask is True.
Cached decode steps (length 1) are left unsteered, so generated tokens see the steered prompt
only through attention to it. Any other length fails: a mask is defined for one exact prompt batch.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch
from jaxtyping import Bool
from torch import Tensor

_MASK: list[Tensor | None] = [None]


def active() -> Tensor | None:
    return _MASK[0]


@contextmanager
def only_tokens(mask: Bool[Tensor, "b s"]) -> Iterator[None]:
    assert _MASK[0] is None, "token masks do not nest"
    assert mask.dtype == torch.bool and mask.ndim == 2
    _MASK[0] = mask
    try:
        yield
    finally:
        _MASK[0] = None


def select(steered: Tensor, original: Tensor, seq_dim: int = 1) -> Tensor:
    """Steered values at masked prompt positions, original values elsewhere and at decode steps."""
    mask = _MASK[0]
    if mask is None:
        return steered
    length = original.shape[seq_dim]
    if length == 1 and mask.shape[1] > 1:
        return original
    assert (original.shape[0], length) == tuple(mask.shape), f"token mask {tuple(mask.shape)} does not match batch/sequence of {tuple(original.shape)}"
    shape = [1] * original.ndim
    shape[0], shape[seq_dim] = mask.shape
    return torch.where(mask.view(shape).to(original.device), steered, original)
