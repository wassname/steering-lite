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


def span_mask(
    input_ids: Int[Tensor, "b s"],
    offsets: Int[Tensor, "b s 2"],
    prompts: list[str],
    spans: list[str],
    tokenizer,
) -> Bool[Tensor, "b s"]:
    """Tokens overlapping each row's first occurrence of its span, including merged whitespace; reject other text. PI/OpenAI."""
    starts = torch.tensor([p.index(span) for p, span in zip(prompts, spans, strict=True)], device=offsets.device)[:, None]
    ends = starts + torch.tensor([len(span) for span in spans], device=offsets.device)[:, None]
    mask = (offsets[..., 1] > starts) & (offsets[..., 0] < ends)
    for ids, selected, span in zip(input_ids, mask, spans, strict=True):
        assert tokenizer.decode(ids[selected]).strip() == span.strip(), "token span includes other text"
    return mask


def instruction_mask(input_ids, offsets, prompts: list[str], instruction: str, tokenizer) -> Bool[Tensor, "b s"]:
    return span_mask(input_ids, offsets, prompts, [instruction] * len(prompts), tokenizer)


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
