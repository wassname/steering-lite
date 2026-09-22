"""VJP-delta in cached value space, using the same target contrast and class difference.

Pullbacks are taken through the actual values returned by DynamicCache.update,
not through a projection-module proxy. Keys and recurrent states are not edited.
Source estimator: https://github.com/wassname/vjp-steering (efcd848).
Implementation: PI/OpenAI.
"""

from dataclasses import dataclass

import torch
from einops import einsum
from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks
from .kv_cache_gram import DynamicCache, DynamicLayer, KVCacheGram, SteeredDynamicCache, _require_dynamic_cache
from .vjp_delta import _activations, _encode, _target_mean, _unit_direction, _valid_mask

_CacheBase = DynamicCache if DynamicCache is not None else object


@register_config
@dataclass
class VjpCacheC(SteeringConfig):
    method: str = "vjp_cache"
    target_layer: int | None = None
    skip_first: int = 16


class ValueGradientCache(_CacheBase):
    def __init__(self, config, selected):
        super().__init__(config=config)
        self.selected = selected
        self.sources = {}

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if layer_idx in self.selected:
            if not isinstance(self.layers[layer_idx], DynamicLayer):
                raise TypeError(f"VJP-cache requires a full-attention DynamicLayer at {layer_idx}")
            # Seed the autograd graph at the earliest selected layer. The model
            # and input parameters are frozen, so the only graph root is a
            # value-cache input: detaching and re-requiring grad on
            # min(selected)'s incoming value seeds the frozen-model autograd
            # graph. DynamicCache.update concatenates storage, so every returned
            # `values` tensor (including the earliest layer's) is a valid
            # non-leaf autograd input, not a leaf. Later selected values stay
            # graph-connected intermediates that autograd.grad accepts
            # directly; we must NOT detach them, because that would sever the
            # legitimate forward influence of earlier selected layers on this
            # layer's cache.
            if layer_idx == min(self.selected):
                value_states = value_states.detach().requires_grad_(True)
        keys, values = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx in self.selected:
            self.sources[layer_idx] = values
        return keys, values


class AdditiveValueCache(SteeredDynamicCache):
    def _edit(self, values, layer_idx):
        if not self._steering_lease.active:
            return values
        directions = self._steering_directions[layer_idx].to(values)
        if directions.shape[1:] != (values.shape[1], values.shape[3]):
            raise ValueError(f"layer {layer_idx}: direction shape does not match cache heads/dim")
        return values + self._steering_coeff * directions.sum(dim=0)[None, :, None, :]


def _cache_gradients(model, tok, prompts, layers, target_layer, cotangent, skip_first, max_length):
    encoded = _encode(model, tok, prompts, max_length)
    valid = _valid_mask(encoded.attention_mask, skip_first)
    if valid.sum(dim=1).min() == 0:
        raise ValueError(f"a prompt has no valid positions after skip_first={skip_first}")
    cache = ValueGradientCache(model.config.get_text_config(), layers)
    with torch.enable_grad(), _activations(model, (target_layer,)) as found:
        model(**encoded, past_key_values=cache, use_cache=True)
        target = found[target_layer]
        sources = [cache.sources[layer] for layer in layers]
        gradients = torch.autograd.grad(
            target, sources,
            grad_outputs=cotangent.detach().to(target)[None, None, :] * valid[:, :, None],
        )
    return dict(zip(layers, gradients, strict=True)), valid


def _class_mean_cache_vjp(model, tok, prompts, layers, target_layer, cotangent, batch_size, max_length, skip_first):
    gradient_totals = {}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        gradients, valid = _cache_gradients(
            model, tok, batch, layers, target_layer, cotangent, skip_first, max_length,
        )
        counts = valid.sum(dim=1).float()
        for layer, gradient in gradients.items():
            per_prompt = einsum(gradient.float(), valid.float(), "b h s d, b s -> b h d") / counts[:, None, None]
            if start == 0:
                gradient_totals[layer] = per_prompt.sum(0)
            else:
                gradient_totals[layer] += per_prompt.sum(0)
    return {layer: total / len(prompts) for layer, total in gradient_totals.items()}


@register
class VjpCache:
    name = "vjp_cache"
    extract_from_prompts = True
    cache_intervention = True

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        _require_dynamic_cache()
        model.requires_grad_(False)
        count = len(_get_blocks(model))
        target = count - 3 if cfg.target_layer is None else cfg.target_layer
        layers = cfg.layers
        if not 0 <= target < count or not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= target:
            raise ValueError("VJP-cache requires unique source layers preceding an in-range target")
        if not pos_prompts or len(pos_prompts) != len(neg_prompts):
            raise ValueError("VJP-cache requires nonempty paired positive/negative prompts")
        cotangent = _target_mean(model, tok, pos_prompts, target, batch_size, max_length) - _target_mean(model, tok, neg_prompts, target, batch_size, max_length)
        positive = _class_mean_cache_vjp(model, tok, pos_prompts, layers, target, cotangent, batch_size, max_length, cfg.skip_first)
        negative = _class_mean_cache_vjp(model, tok, neg_prompts, layers, target, cotangent, batch_size, max_length, cfg.skip_first)
        shapes = {layer: positive[layer].shape for layer in layers}
        directions = {layer: _unit_direction((positive[layer] - negative[layer]).flatten()) for layer in layers}
        return {
            layer: {"shared": {}, "stacked": {"c": direction.reshape(shapes[layer]).unsqueeze(0)}}
            for layer, direction in directions.items()
        }

    @staticmethod
    def install(model, cfg, stacked):
        return KVCacheGram.install(model, cfg, stacked, cache_type=AdditiveValueCache)

    @staticmethod
    def apply(_mod, _x, y, _shared, _stacked, _cfg):
        return y
