"""VJP difference from vjp-steering efcd848 (the reference calls it vjp_delta).

c = mean(h_target_positive) - mean(h_target_negative)
v_layer = mean_positive(J.T @ c) - mean_negative(J.T @ c)

One global sign aligns v with the source activation contrast. Because swapping
both classes leaves the raw difference unchanged, this orientation is a heuristic,
not behavioral validation. Source: https://github.com/wassname/vjp-steering
Adapted to steering-lite registration by PI/OpenAI.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from jaxtyping import Bool, Float, Int
from loguru import logger

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks as _blocks
from ..vector import Vector


@register_config
@dataclass
class VjpDeltaC(SteeringConfig):
    method: str = "vjp_delta"
    target_layer: int | None = None
    skip_first: int = 16


def orient_vjp_delta(directions, activation_axis):
    cosines = {
        layer: torch.nn.functional.cosine_similarity(direction.float().cpu(), activation_axis[layer].float().cpu(), dim=0).item()
        for layer, direction in directions.items()
    }
    score = sum(cosines.values()) / len(cosines)
    if score == 0:
        raise ValueError("VJP difference has zero activation-orientation score")
    return ({layer: -v for layer, v in directions.items()} if score < 0 else directions), cosines, score, score < 0


def _unit_direction(direction: torch.Tensor) -> torch.Tensor:
    norm = direction.float().norm()
    if not torch.isfinite(norm):
        raise ValueError("cannot normalize a nonfinite direction")
    if norm == 0:
        raise ValueError("cannot normalize a zero direction")
    return direction / norm


@contextmanager
def _activations(
    model,
    layers: tuple[int, ...],
    graph_root: int | None = None,
    source_readout: str | None = None,
    target_layer: int | None = None,
):
    found = {}
    handles = []

    def hook(layer):
        def record(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if layer == graph_root:
                hidden = hidden.detach().requires_grad_(True)
                output = (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            found[layer] = hidden
            return output

        return record

    for layer in layers:
        block = _blocks(model)[layer]
        module = block if source_readout is None or layer == target_layer else block.get_submodule(source_readout)
        handles.append(module.register_forward_hook(hook(layer)))
    try:
        yield found
    finally:
        for handle in handles:
            handle.remove()


def _encode(model, tokenizer, prompts: list[str], max_length: int):
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        padding_side="right",
        add_special_tokens=False,
    ).to(next(model.parameters()).device)


def _valid_mask(
    attention_mask: Int[torch.Tensor, "b s"], skip_first: int
) -> Bool[torch.Tensor, "b s"]:
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    real_length = attention_mask.sum(dim=1, keepdim=True)
    return (
        (positions[None, :] >= skip_first)
        & (positions[None, :] < real_length - 1)
        & attention_mask.bool()
    )


def _target_mask(
    attention_mask: Int[torch.Tensor, "b s"], skip_first: int, target_scope: str
) -> Bool[torch.Tensor, "b s"]:
    if target_scope == "all_valid":
        return _valid_mask(attention_mask, skip_first)
    if target_scope == "last_token":
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        mask[rows, attention_mask.sum(dim=1) - 1] = True
        return mask
    raise ValueError(f"unknown target_scope={target_scope!r}")


@torch.no_grad()
def _target_mean(
    model,
    tokenizer,
    prompts: list[str],
    target_layer: int,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    total = None
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        encoded = _encode(model, tokenizer, batch, max_length)
        with _activations(model, (target_layer,)) as found:
            model(**encoded)
        last = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(batch), device=last.device)
        values = found[target_layer][rows, last].float()
        total = values.sum(0) if total is None else total + values.sum(0)
    return total / len(prompts)


def _batch_gradients(
    model,
    tokenizer,
    prompts: list[str],
    layers: tuple[int, ...],
    target_layer: int,
    cotangent: Float[torch.Tensor, " d"],
    skip_first: int,
    max_length: int,
    source_readout: str | None = None,
    target_scope: str = "all_valid",
) -> tuple[
    dict[int, Float[torch.Tensor, "b s d"]],
    Bool[torch.Tensor, "b s"],
    dict[int, Float[torch.Tensor, "b s d"]],
]:
    encoded = _encode(model, tokenizer, prompts, max_length)
    valid = _valid_mask(encoded["attention_mask"], skip_first)
    target_valid = _target_mask(encoded["attention_mask"], skip_first, target_scope)
    if valid.sum(dim=1).min() == 0:
        raise ValueError(f"a prompt has no valid positions after skip_first={skip_first}")

    with _activations(
        model,
        (*layers, target_layer),
        graph_root=min(layers),
        source_readout=source_readout,
        target_layer=target_layer,
    ) as found:
        with torch.enable_grad():
            model(**encoded)
            target = found[target_layer]
            expanded = cotangent.detach().to(target).view(1, 1, -1)
            gradients = torch.autograd.grad(
                target,
                [found[layer] for layer in layers],
                grad_outputs=expanded * target_valid.unsqueeze(-1),
            )
            activations = {layer: found[layer].detach() for layer in layers}
    return dict(zip(layers, gradients, strict=True)), valid, activations


def _class_mean_vjp(
    model,
    tokenizer,
    prompts: list[str],
    layers: tuple[int, ...],
    target_layer: int,
    cotangent: Float[torch.Tensor, " d"],
    batch_size: int,
    max_length: int,
    skip_first: int,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    totals = {layer: torch.zeros_like(cotangent, dtype=torch.float32) for layer in layers}
    activation_totals = {layer: torch.zeros_like(cotangent, dtype=torch.float32) for layer in layers}
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        gradients, valid, activations = _batch_gradients(
            model,
            tokenizer,
            batch,
            layers,
            target_layer,
            cotangent,
            skip_first,
            max_length,
        )
        counts = valid.sum(dim=1, keepdim=True).float()
        for layer, gradient in gradients.items():
            per_prompt = (gradient.float() * valid.unsqueeze(-1)).sum(dim=1) / counts
            totals[layer] += per_prompt.sum(0)
            rows = torch.arange(len(batch), device=valid.device)
            for quantile in (0.25, 0.5, 0.75, 1.0):
                positions = skip_first + ((valid.sum(1) - 1).float() * quantile).long()
                activation_totals[layer] += activations[layer][rows, positions].float().sum(0)
    return (
        {layer: total / len(prompts) for layer, total in totals.items()},
        {layer: total / (4 * len(prompts)) for layer, total in activation_totals.items()},
    )


def vjp_delta(
    model,
    tokenizer,
    positive_prompts: list[str],
    negative_prompts: list[str],
    layers: tuple[int, ...],
    *,
    target_layer: int | None = None,
    batch_size: int = 8,
    max_length: int = 384,
    skip_first: int = 16,
) -> Vector:
    """Extract one normalized VJP-delta direction per source layer."""
    model.requires_grad_(False)
    block_count = len(_blocks(model))
    target_layer = block_count - 3 if target_layer is None else target_layer
    if not 0 <= target_layer < block_count:
        raise ValueError(f"target layer {target_layer} is outside model layers [0, {block_count})")
    if not layers or len(set(layers)) != len(layers) or min(layers) < 0 or max(layers) >= target_layer:
        raise ValueError("source layers must precede the target layer and be nonempty, unique, nonnegative")
    if not positive_prompts or len(positive_prompts) != len(negative_prompts):
        raise ValueError("VJP-delta requires nonempty paired positive/negative prompts")

    cotangent = _target_mean(
        model, tokenizer, positive_prompts, target_layer, batch_size, max_length
    ) - _target_mean(
        model, tokenizer, negative_prompts, target_layer, batch_size, max_length
    )
    positive, positive_activations = _class_mean_vjp(
        model,
        tokenizer,
        positive_prompts,
        layers,
        target_layer,
        cotangent,
        batch_size,
        max_length,
        skip_first,
    )
    negative, negative_activations = _class_mean_vjp(
        model,
        tokenizer,
        negative_prompts,
        layers,
        target_layer,
        cotangent,
        batch_size,
        max_length,
        skip_first,
    )
    directions = {layer: positive[layer] - negative[layer] for layer in layers}
    directions = {layer: _unit_direction(direction) for layer, direction in directions.items()}
    activation_axis = {layer: positive_activations[layer] - negative_activations[layer] for layer in layers}
    for axis in activation_axis.values():
        _unit_direction(axis)
    directions, cosines, score, flipped = orient_vjp_delta(directions, activation_axis)
    if not torch.isfinite(torch.tensor(score)):
        raise ValueError("nonfinite VJP-delta orientation score")
    logger.info("VJP-delta global activation orientation score={} flipped={} cosines={}; heuristic, not behavioral validation", score, flipped, cosines)
    stacked = {layer: {"v": direction.unsqueeze(0)} for layer, direction in directions.items()}
    config = VjpDeltaC(
        layers=layers,
        target_layer=target_layer,
        skip_first=skip_first,
    )
    return Vector(config, {layer: {} for layer in layers}, stacked)


@register
class VjpDelta:
    name = "vjp_delta"
    extract_from_prompts = True

    @staticmethod
    def extract(model, tok, pos_prompts, neg_prompts, cfg, *, batch_size, max_length):
        vector = vjp_delta(
            model, tok, pos_prompts, neg_prompts, cfg.layers,
            target_layer=cfg.target_layer, skip_first=cfg.skip_first,
            batch_size=batch_size, max_length=max_length,
        )
        return {layer: {"shared": vector.shared[layer], "stacked": vector.stacked[layer]} for layer in vector.stacked}

    @staticmethod
    def apply(_mod, _x, y: Float[torch.Tensor, "b s d"], _shared, stacked, cfg):
        return y + cfg.coeff * stacked["v"].to(y).sum(dim=0)
