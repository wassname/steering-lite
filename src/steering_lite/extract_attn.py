"""Attention-weighted activation extraction.

Alternative to last-token pooling. Aggregates hidden states across the prefix
using the model's own attention weights at the last position. Designed to test
hypotheses about *where* the contrastive steering signal lives:

- V hypothesis (content): K/Q routing is roughly shared between (pos, neg)
  pairs (>95% of tokens identical), so use a *shared* attention pattern for
  both halves of the pair. The mean-diff then isolates value-pathway changes:

  $$v_L = \\sum_{i} w^{\\text{shared}}_i \\cdot (h^+_{L,i} - h^-_{L,i})$$

- K/Q hypothesis (routing): use the attention *difference* to highlight
  positions whose routing changed, applied to a shared content reference
  $h^{\\text{ref}} = \\tfrac{1}{2}(h^+ + h^-)$:

  $$v_L = \\sum_{i} (w^+_i - w^-_i) \\cdot h^{\\text{ref}}_{L,i}$$

If we used each sample's own weights for V (mode='v', pair_agg='hdiff'), the
result mixes V-diff and K/Q-diff. The shared-attention modes disentangle them.

Prior art (related but different):
- PASTA https://arxiv.org/abs/2311.02262 -- decode-time attention reweighting,
  not extraction-time aggregation.
- GrAInS https://arxiv.org/abs/2507.18043 -- gradient attribution to weight
  tokens.
- Standard mean/max pooling -- ignores per-position salience.

The pair-aware V/KQ decomposition here is, as far as we know, novel.

Usage: prompts must be **interleaved** [pos_0, neg_0, pos_1, neg_1, ...].
Output is dict[layer_idx, Tensor[2N, d]] of interleaved aggregated states.
Caller splits even/odd back into pos_acts / neg_acts before method.extract.
"""
from __future__ import annotations
from typing import Literal
import torch
from torch import nn, Tensor
from einops import einsum, rearrange
from jaxtyping import Float

from .target import _get_blocks


PairAgg = Literal["mean", "max", "min", "hdiff"]
Mode = Literal["v", "kq"]
Pool = Literal["last", "mean", "attn_v", "attn_kq"]


def _pair_combine(w: Tensor, agg: PairAgg) -> Tensor:
    """Combine attention rows of interleaved (pos, neg) pairs.

    w: [2N, seq, seq]. Returns same shape with each pair (2k, 2k+1) replaced
    by the combined pattern. `hdiff` returns w unchanged (caller uses each
    sample's own weights but applied to h_pos - h_neg).
    """
    if agg == "hdiff":
        return w
    out = w.clone()
    pa = w[0::2]
    pb = w[1::2]
    if agg == "mean":
        c = (pa + pb) / 2
    elif agg == "max":
        c = torch.maximum(pa, pb)
    elif agg == "min":
        c = torch.minimum(pa, pb)
    else:
        raise ValueError(f"unknown pair_agg {agg!r}")
    out[0::2] = c
    out[1::2] = c
    return out


@torch.no_grad()
def record_activations_attn(
    model: nn.Module,
    tok,
    prompts_interleaved: list[str],
    layers: tuple[int, ...],
    *,
    mode: Mode = "v",
    pair_agg: PairAgg = "mean",
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[int, Float[Tensor, "n d"]]:
    """Run prompts (interleaved pos/neg pairs) and aggregate hidden states.

    Returns {layer_idx: Tensor[len(prompts), d]} of aggregated states. The
    length must be even; pairs are (2k, 2k+1).

    Requires `output_attentions=True` to work, which means the model must be
    loaded with `attn_implementation="eager"` (sdpa/flash do not return attn).
    """
    if len(prompts_interleaved) % 2 != 0:
        raise ValueError("prompts_interleaved must have even length (pairs)")
    blocks = _get_blocks(model)
    device = next(model.parameters()).device
    bucket: dict[int, list[Tensor]] = {l: [] for l in layers}

    # forward hook to capture per-block hidden states
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
        for i in range(0, len(prompts_interleaved), batch_size):
            batch = prompts_interleaved[i : i + batch_size]
            # pair alignment: keep pairs together inside a batch
            if (i // batch_size) and (batch_size % 2):
                raise ValueError("batch_size must be even to keep pairs together")
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            mask: Tensor = enc["attention_mask"]
            captured.clear()
            out = model(**enc, output_attentions=True)
            if out.attentions is None:
                raise RuntimeError(
                    "output_attentions=None; load model with "
                    "attn_implementation='eager' (sdpa/flash drop attention)"
                )

            # last non-pad index per row
            last_idx = mask.shape[1] - 1 - mask.flip([-1]).argmax(-1)  # [B]
            B = mask.shape[0]
            if B % 2 != 0:
                raise ValueError("batch produced odd number of rows")

            # per-layer attn at last position, mean over heads -> [B, seq]
            # (we only need attn[b, last_idx[b], :], not full [seq,seq])
            for li in layers:
                # block i in steering-lite is hf model's i-th transformer block
                attn_full: Tensor = out.attentions[li].mean(dim=1).float()  # [B, seq, seq]
                # gather row at last_idx for each b
                row_idx = rearrange(last_idx, "b -> b 1 1").expand(-1, 1, attn_full.size(-1))
                attn_last = attn_full.gather(1, row_idx).squeeze(1)  # [B, seq]
                # zero out padding columns
                attn_last = attn_last * mask.float()

                # apply pair_agg by treating attn_last as [B, 1, seq] for _pair_combine
                attn_last_p = _pair_combine(attn_last.unsqueeze(1), pair_agg).squeeze(1)

                hs: Tensor = captured[li].float()  # [B, seq, d]
                aggregated = []
                for b in range(B):
                    li_b = int(last_idx[b]) + 1
                    if mode == "v":
                        if pair_agg == "hdiff":
                            # this sample's attention applied to canonical pair-diff
                            partner = b + 1 if b % 2 == 0 else b - 1
                            li_min = min(li_b, int(last_idx[partner]) + 1)
                            sign = 1.0 if b % 2 == 0 else -1.0
                            h_diff = sign * (hs[b - (b % 2), :li_min] - hs[b - (b % 2) + 1, :li_min])
                            w = attn_last_p[b, :li_min]
                            aggregated.append(0.5 * einsum(w, h_diff, "s, s d -> d"))
                        else:
                            w = attn_last_p[b, :li_b]
                            h_seq = hs[b, :li_b]
                            aggregated.append(einsum(w, h_seq, "s, s d -> d"))
                    elif mode == "kq":
                        partner = b + 1 if b % 2 == 0 else b - 1
                        li_min = min(li_b, int(last_idx[partner]) + 1)
                        # NOTE: must use *individual* attention rows, not pair-combined
                        delta_w = attn_last[b, :li_min] - attn_last[partner, :li_min]
                        h_ref = (hs[b, :li_min] + hs[partner, :li_min]) / 2
                        aggregated.append(0.5 * einsum(delta_w, h_ref, "s, s d -> d"))
                    else:
                        raise ValueError(f"unknown mode {mode!r}")
                bucket[li].append(torch.stack(aggregated).cpu())

            del out
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {li: torch.cat(bucket[li], dim=0) for li in layers}


@torch.no_grad()
def record_activations_mean(
    model: nn.Module,
    tok,
    prompts: list[str],
    layers: tuple[int, ...],
    *,
    batch_size: int = 8,
    max_length: int = 256,
) -> dict[int, Float[Tensor, "n d"]]:
    """Plain non-pad mean pooling over the prefix. No attention required.

    $$h^{\\text{pool}} = \\tfrac{1}{|\\text{non-pad}|} \\sum_i \\mathbb{1}[m_i] \\, h_i$$
    """
    blocks = _get_blocks(model)
    device = next(model.parameters()).device
    bucket: dict[int, list[Tensor]] = {l: [] for l in layers}
    captured: dict[int, Tensor] = {}

    def make_hook(li: int):
        def hook(_mod, _args, out):
            captured[li] = out[0] if isinstance(out, tuple) else out
        return hook

    handles = [blocks[li].register_forward_hook(make_hook(li)) for li in layers]
    try:
        was_training = model.training
        model.eval()
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length).to(device)
            mask = enc["attention_mask"].float()  # [B, S]
            captured.clear()
            model(**enc)
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
            for li in layers:
                hs = captured[li].float()  # [B, S, d]
                pooled = (hs * mask.unsqueeze(-1)).sum(dim=1) / denom  # [B, d]
                bucket[li].append(pooled.cpu())
        if was_training:
            model.train()
    finally:
        for h in handles:
            h.remove()

    return {li: torch.cat(bucket[li], dim=0) for li in layers}
