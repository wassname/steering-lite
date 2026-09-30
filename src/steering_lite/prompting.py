"""Scale selected prompt-token embeddings before generation. User proposal, 2026-09-30; PI/OpenAI.

E' = E * (1 + (C - 1) M), with a boolean token mask M. C=1 is ordinary
prompting; C=0 leaves zero-valued tokens and their positions, not an empty prompt.
Pass the yielded embeddings to cached HF generation with the original input_ids.
Only prefill uses inputs_embeds; subsequent generated tokens are embedded normally.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import math

import torch
from torch import Tensor, nn
from jaxtyping import Bool, Float, Int


def instruction_mask(
    input_ids: Int[Tensor, "b s"],
    offsets: Int[Tensor, "b s 2"],
    prompts: list[str],
    instruction: str,
    tokenizer,
) -> Bool[Tensor, "b s"]:
    """Select instruction tokens, including merged whitespace; reject other text. PI/OpenAI."""
    starts = torch.tensor([p.index(instruction) for p in prompts], device=offsets.device)[:, None]
    mask = (offsets[..., 1] > starts) & (offsets[..., 0] < starts + len(instruction))
    for ids, selected in zip(input_ids, mask, strict=True):
        assert tokenizer.decode(ids[selected]).strip() == instruction, "instruction token span includes other text"
    return mask


@contextmanager
def scaled_prompt_embeddings(
    model: nn.Module,
    input_ids: Int[Tensor, "b s"],
    mask: Bool[Tensor, "b s"],
    C: float,
) -> Iterator[Float[Tensor, "b s d"]]:
    """Yield scaled inputs; no hooks, weight changes, or persistent model state. PI/OpenAI."""
    assert math.isfinite(C) and C >= 0, "gain must be finite and nonnegative; change the instruction for the opposite persona"
    embeddings = model.get_input_embeddings()(input_ids)
    factors = torch.where(mask, C, 1.0).to(embeddings)
    yield embeddings * factors[..., None]
