"""Contrastive steering of the attention value cache in a Gram basis.

This method changes the persistent value-cache path rather than a block's
residual output. Fit uses actual cached values from positive and negative
prompts. For every selected layer and KV head, it streams prompt-normalized
second moments and class means:

    G = 1/2 E_pos[V^T V / T] + 1/2 E_neg[V^T V / T]
    B_r = top-r eigenvectors(G)
    c = normalize(B_r B_r^T (mean_pos[V] - mean_neg[V]))

The Gram eigenspace is the right-singular subspace of the concatenated cache in
exact arithmetic. It identifies high-energy value directions; the class
contrast selects the direction inside that subspace.

At inference, keys and attention weights at the edited layer are unchanged. New
values are changed before insertion into the cache, while values in a supplied
ordinary prefix cache are changed once when that cache is promoted:

    z_t = V_t c
    V'_t = V_t + coeff |z_t| c

Positive coefficients amplify positive projections and reduce negative ones;
negative coefficients do the reverse. Orthogonal value content is preserved.
No optimizer or backward pass is used.

Selected layers must use transformers full-attention DynamicLayer caches. Hybrid
models work when only their full-attention layers are selected. Promoting an
already-populated hybrid, static, sliding-window, or quantized cache is unsupported.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor, nn

try:
    from transformers.cache_utils import DynamicCache, DynamicLayer
except ImportError:
    DynamicCache = None
    DynamicLayer = None

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks


ε = 1e-8


@register_config
@dataclass
class KVCacheGramC(SteeringConfig):
    method: str = "kv_cache_gram"
    r: int = 16


def _require_dynamic_cache() -> None:
    if DynamicCache is None or DynamicLayer is None:
        raise ImportError("kv_cache_gram requires the steering-lite hf-test extra with transformers 5.x")


@dataclass
class _CacheSteeringLease:
    active: bool = True


_CacheBase = DynamicCache if DynamicCache is not None else object


class SteeredDynamicCache(_CacheBase):
    """DynamicCache that applies one contrastive edit to each incoming value."""

    def __init__(
        self,
        *,
        config,
        directions: dict[int, Tensor],
        coeff: float,
        lease: _CacheSteeringLease,
    ):
        _require_dynamic_cache()
        super().__init__(config=config)
        self._steering_directions = directions
        self._steering_coeff = float(coeff)
        self._steering_lease = lease

    def _edit(self, values: Float[Tensor, "b h t d"], layer_idx: int) -> Float[Tensor, "b h t d"]:
        if not self._steering_lease.active:
            return values
        directions = self._steering_directions.get(layer_idx)
        if directions is None:
            return values
        if values.ndim != 4:
            raise ValueError(
                f"kv_cache_gram expected value cache [b,h,t,d], got {tuple(values.shape)}"
            )
        directions = directions.to(values)
        if directions.shape[1:] != (values.shape[1], values.shape[3]):
            raise ValueError(
                f"layer {layer_idx}: direction heads/dim {tuple(directions.shape[1:])} "
                f"do not match cache {tuple(values.shape[1::2])}"
            )
        magnitude = directions.norm(dim=-1)                              # [k, h]
        direction = directions / (magnitude[..., None] + ε)             # [k, h, d]
        projection = einsum(values, direction, "b h t d, k h d -> b h t k")
        delta = einsum(
            projection.abs() * magnitude.T[None, :, None, :],
            direction,
            "b h t k, k h d -> b h t d",
        )
        return values + self._steering_coeff * delta

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        if layer_idx in self._steering_directions:
            layer = self.layers[layer_idx]
            if type(layer) is not DynamicLayer:
                raise TypeError(
                    f"kv_cache_gram layer {layer_idx} requires DynamicLayer, got {type(layer).__name__}"
                )
            value_states = self._edit(value_states, layer_idx)
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    @classmethod
    def promote(
        cls,
        cache,
        *,
        config,
        directions: dict[int, Tensor],
        coeff: float,
        lease: _CacheSteeringLease,
    ) -> "SteeredDynamicCache":
        """Copy an ordinary cache and edit every existing selected value once."""
        selected_layers = [cache.layers[layer_idx] for layer_idx in directions]
        if all(type(layer) is DynamicLayer and not layer.is_initialized for layer in selected_layers):
            return cls(config=config, directions=directions, coeff=coeff, lease=lease)

        promoted = cls(config=config, directions=directions, coeff=coeff, lease=lease)
        for layer_idx, layer in enumerate(cache.layers):
            if type(layer) is not DynamicLayer:
                raise TypeError(
                    f"kv_cache_gram cannot promote a populated {type(layer).__name__} "
                    f"at layer {layer_idx}"
                )
            if not layer.is_initialized:
                continue
            values = promoted._edit(layer.values, layer_idx)
            super(SteeredDynamicCache, promoted).update(layer.keys, values, layer_idx)
        return promoted

    def matches(self, directions: dict[int, Tensor], coeff: float) -> bool:
        if self._steering_coeff != coeff or self._steering_directions.keys() != directions.keys():
            return False
        return all(
            torch.equal(self._steering_directions[layer_idx].cpu(), directions[layer_idx].cpu())
            for layer_idx in directions
        )


class _CacheHookHandle:
    def __init__(self, hook, lease: _CacheSteeringLease):
        self.hook = hook
        self.lease = lease

    def remove(self) -> None:
        self.lease.active = False
        self.hook.remove()


@register
class KVCacheGram:
    name = "kv_cache_gram"
    cache_intervention = True
    extract_from_prompts = True

    @staticmethod
    def _class_stats(
        model: nn.Module,
        tok,
        prompts: list[str],
        layers: tuple[int, ...],
        *,
        batch_size: int,
        max_length: int,
    ) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
        if not prompts:
            raise ValueError("kv_cache_gram needs at least one prompt in each class")
        device = next(model.parameters()).device
        gram_sum: dict[int, Tensor] = {}
        mean_sum: dict[int, Tensor] = {}
        count = 0
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(prompts), batch_size):
                    enc = tok(
                        prompts[start:start + batch_size],
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    ).to(device)
                    outputs = model(**enc, use_cache=True, return_dict=True)
                    cache = outputs.past_key_values
                    if not isinstance(cache, DynamicCache):
                        raise TypeError(
                            f"kv_cache_gram extraction requires DynamicCache, got {type(cache).__name__}"
                        )
                    mask = enc["attention_mask"].to(torch.bool).cpu()
                    for layer_idx in layers:
                        layer = cache.layers[layer_idx]
                        if type(layer) is not DynamicLayer:
                            raise TypeError(
                                f"kv_cache_gram requires full-attention DynamicLayer, got "
                                f"{type(layer).__name__} at layer {layer_idx}"
                            )
                        values = layer.values.detach().cpu().double()       # [b,h,t,d]
                        for row in range(values.shape[0]):
                            prompt_values = values[row, :, mask[row], :]   # [h,t,d]
                            prompt_mean = prompt_values.mean(dim=1)        # [h,d]
                            prompt_gram = einsum(
                                prompt_values,
                                prompt_values,
                                "h t d, h t e -> h d e",
                            ) / prompt_values.shape[1]
                            if layer_idx not in gram_sum:
                                gram_sum[layer_idx] = prompt_gram
                                mean_sum[layer_idx] = prompt_mean
                            else:
                                gram_sum[layer_idx] += prompt_gram
                                mean_sum[layer_idx] += prompt_mean
                    count += mask.shape[0]
        finally:
            model.train(was_training)
        return (
            {layer_idx: value / count for layer_idx, value in gram_sum.items()},
            {layer_idx: value / count for layer_idx, value in mean_sum.items()},
        )

    @staticmethod
    def extract(
        model: nn.Module,
        tok,
        pos_prompts: list[str],
        neg_prompts: list[str],
        cfg: KVCacheGramC,
        *,
        batch_size: int = 8,
        max_length: int = 256,
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        _require_dynamic_cache()
        blocks = _get_blocks(model)
        layers = tuple(range(len(blocks))) if cfg.layers is None else tuple(cfg.layers)
        if cfg.r == 0 or cfg.r < -1:
            raise ValueError(f"kv_cache_gram r must be -1 or >= 1, got {cfg.r}")
        pos_gram, pos_mean = KVCacheGram._class_stats(
            model, tok, pos_prompts, layers, batch_size=batch_size, max_length=max_length
        )
        neg_gram, neg_mean = KVCacheGram._class_stats(
            model, tok, neg_prompts, layers, batch_size=batch_size, max_length=max_length
        )

        out = {}
        for layer_idx in layers:
            gram = 0.5 * (pos_gram[layer_idx] + neg_gram[layer_idx])       # [h,d,d]
            _, basis = torch.linalg.eigh(gram)                             # ascending
            rank = basis.shape[-1] if cfg.r == -1 else min(cfg.r, basis.shape[-1])
            basis = basis[..., -rank:]                                    # [h,d,r]
            contrast = pos_mean[layer_idx] - neg_mean[layer_idx]          # [h,d]
            coordinates = einsum(contrast, basis, "h d, h d r -> h r")
            direction = einsum(coordinates, basis, "h r, h d r -> h d")
            norm = direction.norm(dim=-1, keepdim=True)
            if not torch.isfinite(norm).all() or torch.any(norm <= ε):
                raise ValueError(
                    f"kv_cache_gram layer {layer_idx} has a non-finite or zero head direction"
                )
            direction = (direction / norm).float().contiguous()
            out[layer_idx] = {"shared": {}, "stacked": {"c": direction.unsqueeze(0)}}
        return out

    @staticmethod
    def install(model: nn.Module, cfg: KVCacheGramC, stacked: dict[int, dict[str, Tensor]]):
        decoder = getattr(model, "model", model)
        if not hasattr(decoder, "layers"):
            language_model = getattr(decoder, "language_model", None)
            decoder = getattr(language_model, "model", language_model)
        if decoder is None or not hasattr(decoder, "layers"):
            raise RuntimeError("kv_cache_gram could not find the decoder module")
        _require_dynamic_cache()
        directions = {layer_idx: state["c"] for layer_idx, state in stacked.items()}
        config = getattr(decoder, "config", model.config)
        lease = _CacheSteeringLease()

        def inject_cache(_module, args, kwargs):
            kwargs["use_cache"] = True
            cache = kwargs.get("past_key_values")
            if cache is None:
                kwargs["past_key_values"] = SteeredDynamicCache(
                    config=config, directions=directions, coeff=cfg.coeff, lease=lease
                )
            elif isinstance(cache, SteeredDynamicCache):
                if not cache.matches(directions, cfg.coeff):
                    raise RuntimeError("past_key_values belongs to a different kv_cache_gram attachment")
                cache._steering_directions = directions
                cache._steering_lease = lease
            elif isinstance(cache, DynamicCache):
                kwargs["past_key_values"] = SteeredDynamicCache.promote(
                    cache, config=config, directions=directions, coeff=cfg.coeff, lease=lease
                )
            else:
                raise TypeError(
                    f"kv_cache_gram requires DynamicCache, got {type(cache).__name__}"
                )
            return args, kwargs

        hook = decoder.register_forward_pre_hook(inject_cache, with_kwargs=True)
        return [_CacheHookHandle(hook, lease)]
