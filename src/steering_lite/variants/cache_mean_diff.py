"""One-shot mean-difference steering of the final prompt token's value cache.

Belitsky et al. 2025: https://arxiv.org/abs/2507.08799 (values-only setting).
For each full-attention layer, extract c = mean(V_pos[last]) - mean(V_neg[last]).
After the complete prompt forward, edit V_prompt[last] += coeff * c once.
Later tokens attend to this edited memory; their incoming values are unchanged.

Unlike the reference's offset-token generation, this variant keeps the prompt
unchanged. The first output token is unsteered; the effect starts at token two.
Calibration must score continuations with the edited prompt cache, not re-prefill
prompt and answer together. Ordinary populated hybrid caches cannot be promoted.
Implementation: PI/OpenAI.
"""
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from .. import positions
from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks
from .value_gram import DynamicCache, DynamicLayer, SteeredDynamicCache, ValueGram, _require_dynamic_cache
from .vjp_resid import _encode


@register_config
@dataclass
class CacheMeanDiffC(SteeringConfig):
    method: str = "cache_mean_diff"


def _last_indices(mask: Bool[Tensor, "b s"]) -> Tensor:
    assert mask.any(dim=1).all(), "cache_mean_diff needs a real prompt token in every row"
    return torch.arange(mask.shape[1], device=mask.device).expand_as(mask).masked_fill(~mask, -1).max(-1).values


class PromptValueCache(SteeredDynamicCache):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.edited = False

    def _edit(self, values: Float[Tensor, "b h s d"], layer_idx: int) -> Float[Tensor, "b h s d"]:
        return values

    def edit_prompt(self, mask: Bool[Tensor, "b s"] | None = None) -> None:
        if self.edited or not self._steering_lease.active:
            return
        assert positions.active() is None, "cache_mean_diff edits the final prompt token; it has no per-token form"
        for layer_idx, directions in self._steering_directions.items():
            layer = self.layers[layer_idx]
            if type(layer) is not DynamicLayer:
                raise TypeError(f"cache_mean_diff needs a full-attention DynamicLayer at {layer_idx}")
            values = layer.values.clone()
            rows = torch.arange(values.shape[0], device=values.device)
            last = values.shape[2] - 1 if mask is None else _last_indices(mask.to(values.device))
            values[rows, :, last, :] += self._steering_coeff * directions.to(values).sum(0)
            layer.values = values
        self.edited = True

    @classmethod
    def promote(cls, cache, **kwargs):
        populated = cache.get_seq_length() > 0
        promoted = super().promote(cache, **kwargs)
        if populated:
            promoted.edit_prompt()
        return promoted


@register
class CacheMeanDiff:
    name = "cache_mean_diff"
    extract_from_prompts = True
    cache_intervention = True
    prompt_cache_only = True

    @staticmethod
    @torch.no_grad()
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        _require_dynamic_cache()
        if not pos_prompts or len(pos_prompts) != len(neg_prompts):
            raise ValueError("cache_mean_diff needs nonempty paired positive/negative prompts")
        if cfg.layers is None:
            blocks = _get_blocks(model)
            types = getattr(model.config.get_text_config(), "layer_types", None)
            cfg.layers = tuple(i for i in range(len(blocks)) if types is None or types[i] == "full_attention")
        if not cfg.layers:
            raise ValueError("cache_mean_diff needs at least one full-attention layer")
        contrast = {}
        for sign, prompts in ((1.0, pos_prompts), (-1.0, neg_prompts)):
            for start in range(0, len(prompts), batch_size):
                batch = _encode(model, tok, prompts[start:start + batch_size], max_length)
                cache = model(**batch, use_cache=True).past_key_values
                last = _last_indices(batch.attention_mask.bool())
                rows = torch.arange(last.shape[0], device=last.device)
                for layer_idx in cfg.layers:
                    layer = cache.layers[layer_idx]
                    if type(layer) is not DynamicLayer:
                        raise TypeError(f"cache_mean_diff needs a full-attention DynamicLayer at {layer_idx}")
                    delta = sign * layer.values[rows, :, last, :].float().sum(0).cpu() / len(prompts)
                    if layer_idx not in contrast:
                        contrast[layer_idx] = delta
                    else:
                        contrast[layer_idx] += delta
        if not all(torch.isfinite(c).all() for c in contrast.values()) or not any(c.norm() > 0 for c in contrast.values()):
            raise ValueError("cache_mean_diff extracted a non-finite or all-zero contrast")
        return {layer: {"shared": {}, "stacked": {"c": c.unsqueeze(0)}} for layer, c in contrast.items()}

    @staticmethod
    def install(model, cfg, stacked):
        handles = ValueGram.install(model, cfg, stacked, cache_type=PromptValueCache)

        def finish_prefill(_module, args, kwargs, output):
            cache = output.past_key_values
            if type(cache) is not PromptValueCache:
                raise TypeError("cache_mean_diff requires a returned PromptValueCache")
            mask = kwargs.get("attention_mask")
            cache.edit_prompt(None if mask is None else mask.bool())

        handles.append(model.register_forward_hook(finish_prefill, with_kwargs=True))
        return handles

    @staticmethod
    def score_continuation(
        model,
        prompt: Int[Tensor, "b s"],
        generated: Int[Tensor, "b t"],
    ) -> Float[Tensor, "b t v"]:
        prefix = model(prompt, use_cache=True)
        logits = prefix.logits[:, -1:]
        if generated.shape[1] > 1:
            continuation = model(
                generated[:, :-1], past_key_values=prefix.past_key_values, use_cache=True,
            )
            logits = torch.cat([logits, continuation.logits], dim=1)
        return logits

    @staticmethod
    def apply(_mod, _x, y, _shared, _stacked, _cfg):
        return y
