"""Mean VJP: vjp_delta's pullback J.T @ c, averaged over prompts instead of differenced by persona.

c = mean(h_target | positive) - mean(h_target | negative)           # same cotangent as vjp_delta
vjp_delta:     v_l = mean_positive(J_l.T @ c) - mean_negative(J_l.T @ c)
mean_vjp:      v_l = mean_{positive + negative}(J_l.T @ c)
wiki_mean_vjp: v_l = mean_{WikiText}(J_l.T @ c)                      # contexts cut to the persona prompts' lengths

Intuition: J_l.T @ c is the residual change at layer l that most increases the target-layer
contrast c. vjp_delta keeps only the part that differs between the two personas; mean_vjp keeps the
average pullback, and wiki_mean_vjp estimates it on generic text, so the direction does not depend
on the persona prompts' own context. Idea #1 in j-steer-dev slop/experiments/mean-vjp (PI/OpenAI).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from jaxtyping import Float

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks as _blocks
from .vjp_delta import _class_mean_vjp, _target_mean, _unit_direction

WIKITEXT = Path(__file__).resolve().parents[1] / "data/wikitext_contexts.json"


@register_config
@dataclass
class MeanVjpC(SteeringConfig):
    method: str = "mean_vjp"
    target_layer: int | None = None
    skip_first: int = 16


@register_config
@dataclass
class WikiMeanVjpC(SteeringConfig):
    method: str = "wiki_mean_vjp"
    target_layer: int | None = None
    skip_first: int = 16


def _wikitext_like(tokenizer, prompts: list[str], max_length: int) -> list[str]:
    """One WikiText context per prompt, cut to that prompt's token length (same positions, same compute)."""
    contexts = json.loads(WIKITEXT.read_text())["contexts"]
    assert len(contexts) >= len(prompts), f"{len(contexts)} WikiText contexts < {len(prompts)} prompts"
    out = []
    for prompt, context in zip(prompts, contexts):
        length = min(len(tokenizer(prompt).input_ids), max_length)
        ids = tokenizer(context).input_ids
        assert len(ids) >= length, f"WikiText context has {len(ids)} tokens < {length}"
        out.append(tokenizer.decode(ids[:length]))
    return out


def _extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length, corpus: str):
    model.requires_grad_(False)
    block_count = len(_blocks(model))
    target_layer = block_count - 3 if cfg.target_layer is None else cfg.target_layer
    layers = cfg.layers
    if not layers or min(layers) < 0 or max(layers) >= target_layer:
        raise ValueError(f"source layers {layers} must precede target layer {target_layer}")
    cotangent = _target_mean(model, tok, pos_prompts, target_layer, batch_size, max_length) - _target_mean(
        model, tok, neg_prompts, target_layer, batch_size, max_length
    )
    prompts = pos_prompts + neg_prompts
    if corpus == "wikitext":
        prompts = _wikitext_like(tok, prompts, max_length)
    mean = _class_mean_vjp(
        model, tok, prompts, layers, target_layer, cotangent, batch_size, max_length, cfg.skip_first
    )
    cfg.target_layer = target_layer
    return {layer: {"shared": {}, "stacked": {"v": _unit_direction(mean[layer]).unsqueeze(0)}} for layer in layers}


def _apply(_mod, _x, y: Float[torch.Tensor, "b s d"], _shared, stacked, cfg):
    return y + cfg.coeff * stacked["v"].to(y).sum(dim=0)


@register
class MeanVjp:
    name = "mean_vjp"
    extract_from_prompts = True

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        return _extract(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length, corpus="persona")

    apply = staticmethod(_apply)


@register
class WikiMeanVjp:
    name = "wiki_mean_vjp"
    extract_from_prompts = True

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        return _extract(model, tok, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length, corpus="wikitext")

    apply = staticmethod(_apply)
