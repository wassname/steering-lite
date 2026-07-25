"""Decode steering directions into associated words with the model's unembedding.

This ports the newer prompt-free readout from wassname/j-steer-dev:
https://github.com/wassname/j-steer-dev

Ported by Codex. The Super S-space reconstruction is a SteeringLike extension
so the readout sees its applied residual direction rather than S coordinates.
"""
from __future__ import annotations

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


def _residual_directions(v, layer, name, tensor, d_model, stacked):
    """Every readable [d_model] row this tensor carries, as (suffix, direction) pairs.

    Half the registry keeps its direction in `shared` rather than `stacked` (pca, spherical,
    cosine_gated and directional_ablation all use shared['v']), and some carry several directions at
    once (topk_clusters' C is one row per cluster, angular_steering's b1/b2 span a plane). Square
    matrices in `shared` are bases, not directions, so they stay out.
    """
    direction = tensor.sum(0) if stacked else tensor
    shared = v.shared[layer]
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
def readout_words(model, tok, v, k=8) -> dict:
    """Logit-lens each readable vector direction into its associated word tokens.

    Top-k words are associated with the +v pole and bottom-k words with -v. This
    is a property of the vector and model only: there is no prompt, axis rubric,
    coefficient, stem, or supplied word list.
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
                    normalized = final_norm(direction.to(W_U.dtype).to(W_U.device))
                    logits = W_U[:vocab].float() @ normalized.float()

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
