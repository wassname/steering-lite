"""Token masks for prompt spans (used by user-turn steering). PI/OpenAI."""

import torch
from torch import Tensor
from jaxtyping import Bool, Int


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
