"""Decode steering directions into associated words with the model's unembedding.

This ports the newer prompt-free readout from wassname/j-steer-dev:
https://github.com/wassname/j-steer-dev

Ported by Codex. The Super S-space reconstruction is a SteeringLike extension
so the readout sees its applied residual direction rather than S coordinates.
"""
from __future__ import annotations

import contextlib

import torch

from .tuned_lens import IdentityLens


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


@torch.no_grad()
def midpoint_states(model, tok, pos_prompts, neg_prompts, layers, *, batch_size=8, max_length=384):
    """Per-PAIR contrast midpoints, {layer: [n_pairs, d]}, the base point a readout reads at.

    `make_persona_pairs` pairs one-to-one: same user message, same suffix, different persona only.
    So (h_pos_i + h_neg_i)/2 differs from a state the model really visits only in the persona
    clause, and it sits between the two poles by construction.

    One row per PAIR rather than one mean over all of them, because the decoder is nonlinear:
    unembed(mean(h)) is not mean(unembed(h)). Average after decoding instead, which
    `readout_at_point` does.
    """
    from jlens.hooks import ActivationRecorder

    def last_states(prompts):
        out = {layer: [] for layer in layers}
        for start in range(0, len(prompts), batch_size):
            enc = tok(prompts[start:start + batch_size], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_length, padding_side="left",
                      add_special_tokens=False).to(model.device)
            with ActivationRecorder(model.model.layers, at=list(layers)) as rec:
                model(**enc)
            for layer in layers:
                out[layer].append(rec.activations[layer][:, -1].float().detach().cpu())
        return {layer: torch.cat(rows) for layer, rows in out.items()}

    if len(pos_prompts) != len(neg_prompts):
        raise ValueError(f"pairs must be one-to-one: {len(pos_prompts)} pos, {len(neg_prompts)} neg")
    pos, neg = last_states(pos_prompts), last_states(neg_prompts)
    return {layer: 0.5 * (pos[layer] + neg[layer]) for layer in layers}


@torch.no_grad()
def readout_at_point(model, tok, v, hs_mid, *, coeff, lens=None, k=8, apply_fn=None) -> dict:
    """Words the vector promotes and suppresses at a real operating point, in nats.

    A steering vector is a displacement, not a state, so it has no words of its own. The decoder
    is `unembed(x) = lm_head(final_norm(x))`, and `final_norm` divides by x's own magnitude, so
    feeding it a bare direction ranks whichever token embeddings are outliers. That is how a
    readout ends up looking like an "explicit content" axis whatever the vector does.

    So read TWO states and difference them, based at the contrast midpoint:

        gain(t) = logsoftmax(unembed(lens(hs_mid + C v)))_t - logsoftmax(unembed(lens(hs_mid)))_t

    where `hs_mid` is the per-pair midpoint of the contrast set the vector came from, and +C and
    -C land on the two poles: `hs_mid + C v` toward positive, `hs_mid - C v` toward negative.
    The model's own `final_norm` is applied here rather than folded into the lens, because it is
    the only nonlinearity in the path: without it the base point cancels algebraically and you
    are back to reading a bare direction.

    Two knobs decide whether the output means anything.

    `coeff` is a READING dose and is not the steering dose. At the calibrated breakdown dose the
    probe point sits far from anywhere the model visits, and the words revert to outlier junk;
    measured on Qwen3.5-4B at ~3 nats/token, every method read as NSFW and multilingual
    fragments. Start well below it and raise until the words move.

    `lens` maps a mid-layer state into final-layer coordinates: TunedLens (fitted, any model),
    JacobianLens (published, two models), or None for the plain logit lens, which is honest only
    in the last few layers.
    """
    W_U, final_norm = model.lm_head.weight, model.model.norm
    vocab = min(len(tok), W_U.shape[0])
    d_model = W_U.shape[1]
    lens = lens or IdentityLens()
    apply_fn = apply_fn or (lambda direction, layer, h, c: h + c * direction.to(h.device, h.dtype))

    def _clean(token_id):
        text = tok.decode([token_id], clean_up_tokenization_spaces=False)
        return is_word_token(text) and text.strip().isascii() and text.strip().isalpha()

    word_mask = torch.tensor([_clean(t) for t in range(vocab)], device=W_U.device)

    def _logprobs(hidden, at_layer):
        """[n, vocab] log-probs for a batch of states, decoded one state at a time."""
        predicted = lens.at_point(hidden.float().cpu(), at_layer).to(W_U.device)
        logits = final_norm(predicted.to(W_U.dtype)).float() @ W_U[:vocab].float().T
        return torch.log_softmax(logits, dim=-1)

    def _top(delta, largest, is_shift=True):
        """Top-k words by MEAN value, with the fraction of base points agreeing on the sign.

        `is_shift=False` for the unsteered base row, whose values are log-probs rather than
        shifts. Sign agreement is meaningless there, since every log-prob is negative, so it
        reports agreement on being in this row's own top-k instead.
        """
        mean = delta.mean(0)
        fill = float("-inf") if largest else float("inf")
        idx = mean.masked_fill(~word_mask, fill).topk(k, largest=largest).indices
        if is_shift:
            agree = (delta > 0).float().mean(0)
        else:
            per_row_top = delta.masked_fill(~word_mask, fill).topk(k, largest=largest, dim=-1).indices
            agree = torch.zeros_like(mean)
            for row in per_row_top:
                agree[row] += 1.0 / delta.shape[0]
        return [(tok.decode([i]).strip(), round(mean[i].item(), 2), round(agree[i].item(), 2))
                for i in idx.tolist()]

    layers, skipped = {}, []
    for stacked, tree in ((True, v.stacked), (False, v.shared)):
        for key in sorted(k2 for k2 in tree if isinstance(k2, int)):
            if key not in hs_mid:
                continue
            for name, tensor in tree[key].items():
                found = _residual_directions(v, key, name, tensor, d_model, stacked)
                if not found:
                    skipped.append(f"{key}.{name}{list(tensor.shape)}")
                    continue
                for suffix, direction in found:
                    base = hs_mid[key]
                    lp_base = _logprobs(base, key)
                    lp_pos = _logprobs(apply_fn(direction, key, base, +coeff), key)
                    lp_neg = _logprobs(apply_fn(direction, key, base, -coeff), key)
                    layers[f"{key}.{name}{suffix}"] = {
                        "base_top": _top(lp_base, True, is_shift=False),
                        "pos_promotes": _top(lp_pos - lp_base, True),
                        "pos_removes": _top(lp_pos - lp_base, False),
                        "neg_promotes": _top(lp_neg - lp_base, True),
                        "max_gain_nats": float((lp_pos - lp_base).mean(0).max()),
                    }
    if not layers:
        raise ValueError(f"{v.cfg.method!r} has no readable direction at these layers: {skipped}")
    return {"method": v.cfg.method, "coeff": coeff, "lens": type(lens).__name__,
            "n_base": len(next(iter(hs_mid.values()))), "layers": layers, "skipped": skipped}


def format_at_point(readout: dict) -> str:
    def words(entries):
        return "  ".join(f"{w}({d:+.2f},{a:.0%})" for w, d, a in entries)

    rows = "\n".join(
        f"  {key:<12} base             : {words(row['base_top'])}\n"
        f"  {'':<12} +C makes likelier: {words(row['pos_promotes'])}"
        f"  (max {row['max_gain_nats']:+.2f} nats)\n"
        f"  {'':<12} +C makes rarer   : {words(row['pos_removes'])}\n"
        f"  {'':<12} -C makes likelier: {words(row['neg_promotes'])}"
        for key, row in readout["layers"].items()
    )
    skipped = (f"\n  (no readable direction for {readout['skipped']})"
               if readout["skipped"] else "")
    return (
        f"VECTOR READOUT AT A BASE POINT ({readout['method']}) C={readout['coeff']:+.4f} "
        f"lens={readout['lens']} over {readout['n_base']} contrast midpoints\n"
        "Change in log-prob at the midpoint of the contrast set, in nats. Each entry is\n"
        "word(mean shift, fraction of base points agreeing on the sign).\n"
        "SHOULD: +C and -C promote OPPOSING fields, and both differ from the base row.\n"
        "ELSE, two named failures. +C and -C promote the SAME words: the direction is\n"
        "unsigned, so it is a magnitude effect and not an axis. Subword fragments, or NSFW\n"
        "and multilingual junk: read the OTHER layers before you judge the vector. Each\n"
        "method reads over its own band and is junk outside it (measured on Qwen3.5-4B\n"
        "layers 6-24: mean_diff gives English over the whole range, vjp_stem only over about\n"
        "16-20). Junk at every layer means the lens is not transporting -- swap it.\n"
        "Do NOT reach for the coeff to fix junk. Over 0.1x to 1.0x of the calibrated ceiling\n"
        "the top words do not move and the gain scales linearly, so that whole span is one\n"
        "linear regime; coeff sets the nats, not which words you see.\n"
        "Agreement below ~80% means the word comes from a few base points, not the direction.\n"
        + rows + skipped
    )
