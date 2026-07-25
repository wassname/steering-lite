"""Decode steering directions into associated words with the model's unembedding.

This ports the newer prompt-free readout from wassname/j-steer-dev:
https://github.com/wassname/j-steer-dev

Ported by Codex. The Super S-space reconstruction is a SteeringLike extension
so the readout sees its applied residual direction rather than S coordinates.
"""
from __future__ import annotations

import contextlib

import torch


def is_word_token(text: str) -> bool:
    stripped = text.strip()
    if not stripped or "<|" in stripped or (
        stripped.startswith("<") and stripped.endswith(">")
    ):
        return False
    return all(
        char.isalnum() or (0 < position < len(stripped) - 1 and char in "'-")
        for position, char in enumerate(stripped)
    )


# Entries in `shared` that are [d_model] but are not residual directions.
NOT_DIRECTIONS = {"sqrtS", "sigma", "p"}


def _residual_directions(v, layer, name, tensor, d_model, stacked):
    """Every readable [d_model] row this tensor carries, as (suffix, direction) pairs.

    Half the registry keeps its direction in `shared` rather than `stacked` (pca, spherical,
    cosine_gated and directional_ablation all use shared['v']), and some carry several directions at
    once (topk_clusters' C is one row per cluster, angular_steering's b1/b2 span a plane). Square
    matrices in `shared` are bases, not directions, so they stay out.
    """
    # sqrtS is a per-coordinate scale in S-space, not a direction in residual space, so lensing it
    # produces words that mean nothing. Shape alone cannot tell the two apart: both are [d_model].
    if not stacked and name in NOT_DIRECTIONS:
        return []
    direction = tensor.sum(0) if stacked else tensor
    # A layer with no shared parts is absent after a save/load round-trip, present as {} when freshly
    # built. Only the sspace family reads this at all.
    shared = v.shared.get(layer, {})
    if name == "dS" and {"U_r", "sqrtS"} <= set(shared):
        direction = (direction * shared["sqrtS"]) @ shared["U_r"].T
    if direction.ndim == 1 and direction.numel() == d_model:
        return [("", direction)]
    # A matrix with as many rows as the model has dimensions spans the space, so it is a basis
    # (U_r, V_corda) and reading its rows as directions would just be noise. Fewer rows than that
    # means a short list of real directions: CHaRS's a and b, topk_clusters' C.
    if direction.ndim == 2 and direction.shape[-1] == d_model and direction.shape[0] < d_model:
        return [(f"[{i}]", row) for i, row in enumerate(direction)]
    return []


@torch.no_grad()
def readout_words(model, tok, v, k=8, lens=None) -> dict:
    """Lens each readable vector direction into its associated word tokens.

    Top-k words are associated with the +v pole and bottom-k words with -v. This
    is a property of the vector and model only: there is no prompt, axis rubric,
    coefficient, stem, or supplied word list.

    Pass a fitted `TunedLens` whenever the vector lives below the last few layers. Without one this
    falls back to the plain logit lens, which cannot read a layer-12 direction at all: on
    Qwen2.5-7B the real hidden state answering ' Tokyo' lenses to Chinese boilerplate until layer 25.
    """
    W_U, final_norm = model.lm_head.weight, model.model.norm
    vocab = min(len(tok), W_U.shape[0])
    d_model = W_U.shape[1]

    def _clean(token_id):
        text = tok.decode([token_id], clean_up_tokenization_spaces=False)
        return is_word_token(text) and text.strip().isascii() and text.strip().isalpha()

    word_mask = torch.tensor(
        [_clean(token_id) for token_id in range(vocab)],
        device=W_U.device,
    )

    def _depth(key):
        text = str(key)
        return (0, int(text), "") if text.isdigit() else (1, 0, text)

    layers, skipped = {}, []
    for stacked, tree in ((True, v.stacked), (False, v.shared)):
        for layer in sorted(tree, key=_depth):
            for name, tensor in tree[layer].items():
                found = _residual_directions(v, layer, name, tensor, d_model, stacked)
                if not found:
                    skipped.append(f"{layer}.{name}{list(tensor.shape)}")
                    continue
                for suffix, direction in found:
                    if lens is not None and isinstance(layer, int):
                        # A already lands in the normalised final space the unembedding reads, and
                        # RMSNorm of a difference of hidden states means nothing, so do not re-norm.
                        readable = lens.to_final(direction.float().cpu(), layer).to(W_U.device)
                    else:
                        readable = final_norm(direction.to(W_U.dtype).to(W_U.device))
                    logits = W_U[:vocab].float() @ readable.float()

                    def top_words(largest):
                        fill = float("-inf") if largest else float("inf")
                        indices = logits.masked_fill(~word_mask, fill).topk(
                            k,
                            largest=largest,
                        ).indices.tolist()
                        return [tok.decode([token_id]).strip() for token_id in indices]

                    layers[f"{layer}.{name}{suffix}"] = {
                        "pos": top_words(True),
                        "neg": top_words(False),
                        "norm": direction.norm().item(),
                    }
    if not layers:
        raise ValueError(f"{v.cfg.method!r} has no residual direction to read: {skipped}")
    return {"method": v.cfg.method, "layers": layers, "skipped": skipped}


def format_readout(readout: dict) -> str:
    rows = "\n".join(
        f"  {key:<14} |v|={row['norm']:7.3f}  +v: {row['pos']}\n"
        f"  {'':<14}              -v: {row['neg']}"
        for key, row in readout["layers"].items()
    )
    skipped = (
        f"\n  (no residual readout for {readout['skipped']})"
        if readout["skipped"]
        else ""
    )
    return (
        f"VECTOR READOUT ({readout['method']}) -- words associated with +v and -v "
        "through final_norm + unembedding; no prompt, stem, or axis word list\n"
        "SHOULD: the +v and -v lists are coherent words and read as opposites of each other.\n"
        "Punctuation/fragments or the two sides looking alike = this direction is not aligned\n"
        "with the unembedding AT THAT LAYER (common in early layers, and expected for\n"
        "erase-only delivery where the applied delta is data-dependent). That is a fact about\n"
        "the logit lens at that depth, NOT evidence that the vector is weak: judge the vector\n"
        "by its effect on the output distribution instead.\n"
        + rows
        + skipped
    )


@torch.no_grad()
def _final_logprobs(model, tok, W_U, final_norm, vocab, prompt, ctx):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(W_U.device)
    with ctx:
        h = model(**enc, output_hidden_states=True).hidden_states[-1][0, -1]
    logits = W_U[:vocab].float() @ final_norm(h.to(W_U.dtype)).float()
    return torch.log_softmax(logits, dim=-1)


@torch.no_grad()
def readout_effect(model, tok, v, c_pos, c_neg, stems, k=8) -> dict:
    """What each pole does to the OUTPUT distribution, in nats, summed over stems.

    Restores the pre-6425e33 j-steer-dev readout. A vector built at layer 12 is not readable by
    logit-lensing the direction itself, because the unembedding cannot read the residual stream at
    that depth (measured on Qwen2.5-7B: the hidden state that answers ' Tokyo' lenses to Chinese
    boilerplate until layer 25). Applying the vector and reading the final layer lets layers 13-28
    do their job first, and the bare-subtracted difference is a log ratio, so it reports what the
    steering changed rather than what the model says most often.

    `stems` are supplied by the caller and printed in the output, because what a vector promotes is
    only defined relative to what it was asked. Keep them generic: axis-specific stems are the knob
    that made a sycophancy run print authority words.
    """
    W_U, final_norm = model.lm_head.weight, model.model.norm
    vocab = min(len(tok), W_U.shape[0])

    def _clean(token_id):
        text = tok.decode([token_id], clean_up_tokenization_spaces=False)
        return is_word_token(text) and text.strip().isascii() and text.strip().isalpha()

    word_mask = torch.tensor([_clean(t) for t in range(vocab)], device=W_U.device)
    pos_dlogp = torch.zeros(vocab, device=W_U.device)
    neg_dlogp = torch.zeros(vocab, device=W_U.device)
    bare_abs = torch.zeros(vocab, device=W_U.device)

    for stem in stems:
        lp_bare = _final_logprobs(model, tok, W_U, final_norm, vocab, stem, contextlib.nullcontext())
        lp_pos = _final_logprobs(model, tok, W_U, final_norm, vocab, stem, v(model, C=+c_pos))
        lp_neg = _final_logprobs(model, tok, W_U, final_norm, vocab, stem, v(model, C=-abs(c_neg)))
        pos_dlogp += lp_pos - lp_bare
        neg_dlogp += lp_neg - lp_bare
        bare_abs += lp_bare

    def _top(scores, largest, n=k):
        fill = float("-inf") if largest else float("inf")
        idx = scores.masked_fill(~word_mask, fill).topk(n, largest=largest).indices.tolist()
        return [tok.decode([i]).strip() for i in idx]

    return {
        "method": v.cfg.method,
        "c_pos": c_pos,
        "c_neg": c_neg,
        "n_stems": len(stems),
        "stems": list(stems),
        "bare": _top(bare_abs, True),
        "promotes": {"pos": _top(pos_dlogp, True), "neg": _top(neg_dlogp, True)},
        "removes": {"pos": _top(pos_dlogp, False), "neg": _top(neg_dlogp, False)},
    }


def format_effect(readout: dict) -> str:
    return (
        f"VECTOR EFFECT READOUT ({readout['method']}, read +{readout['c_pos']:.3f}/"
        f"-{abs(readout['c_neg']):.3f} over {readout['n_stems']} stems)\n"
        "Change in final-layer logprob vs unsteered, in nats, summed over stems.\n"
        "SHOULD: the two poles promote opposing word fields, and neither matches the unsteered\n"
        "top-k. ELSE the poles agree = unsigned perturbation, or they match bare = no effect.\n"
        f"  unsteered  : {readout['bare']}\n"
        f"  +C promotes: {readout['promotes']['pos']}\n"
        f"  -C promotes: {readout['promotes']['neg']}\n"
        f"  +C removes : {readout['removes']['pos']}\n"
        f"  -C removes : {readout['removes']['neg']}"
    )
