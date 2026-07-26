r"""Wassname's Jacobian-Lens steering method.

This adapts Anthropic's public `jacobian-lens` project to activation steering:
https://github.com/anthropics/jacobian-lens

The public reference implementation lives in `wassname/j-steer-dev`:
https://github.com/wassname/j-steer-dev/blob/main/src/jsteer/variants/vjp.py

Start with the ordinary contrastive vector used in activation steering, measured
at target layer $T$. Treat it as a cotangent:

$$c = \bar h_T^+ - \bar h_T^-.$$

For each requested source layer $L$, autograd computes the vector-Jacobian
product $J_{L \to T}(x)^\top c$. This is the pullback of the target contrast
to layer $L$: it identifies source-layer changes that locally produce movement
along $c$ at the target. The full Jacobian is never materialized. We compute
pullbacks for both prompt classes, then subtract their class means:

$$\tilde v_L = \mathbb E_{x^+}[J_{L \to T}(x^+)^\top c]
             - \mathbb E_{x^-}[J_{L \to T}(x^-)^\top c].$$

This raw separation gradient is unchanged if the two class labels are swapped.
We therefore choose one global sign across all source layers so that its mean
layerwise cosine with $\bar h_L^+ - \bar h_L^-$ is positive. The resulting
$v_L$ is normalized and used as the steering direction. This port preserves
j-steer's prompt masks and reductions. Local forward hooks perform the same VJP
without adding `jacobian-lens` as a runtime dependency.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from loguru import logger
from tabulate import tabulate
from torch import Tensor, nn
from tqdm.auto import tqdm

from ..config import SteeringConfig, register, register_config
from ..target import _get_blocks


ε = 1e-8
DEFAULT_TARGET_LAYER_FROM_END = 3


@register_config
@dataclass
class VjpDeltaC(SteeringConfig):
    method: str = "vjp_delta"
    target_layer: int | None = None
    skip_first: int = 16
    cotangent_scope: Literal["all_valid", "last_token"] = "all_valid"
    source_scope: Literal["all_valid", "last_token"] = "all_valid"
    normalize: bool = True
    apply_mode: Literal["add", "damp_amp"] = "add"


@contextmanager
def _record_activations(
    blocks: nn.ModuleList,
    layers: tuple[int, ...],
    *,
    graph_root: int | None = None,
):
    activations: dict[int, Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            if layer == graph_root:
                hidden.requires_grad_(True)
            activations[layer] = hidden

        return hook

    try:
        for layer in sorted(set(layers) | ({graph_root} if graph_root is not None else set())):
            handles.append(blocks[layer].register_forward_hook(make_hook(layer)))
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _unit_vector(vector: Tensor) -> Tensor:
    norm = vector.float().norm()
    if not torch.isfinite(norm).item():
        raise ValueError("cannot normalize a nonfinite vjp_delta direction")
    if norm.item() == 0:
        raise ValueError("cannot normalize a zero vjp_delta direction")
    return vector / norm.to(vector)


def orient_vjp_delta(
    per_layer: dict[int, Tensor],
    activation_axis: dict[int, Tensor],
) -> tuple[dict[int, Tensor], dict[int, float], float, bool]:
    raw_axis_cosines = {
        layer: F.cosine_similarity(
            per_layer[layer].float().cpu(),
            activation_axis[layer].float().cpu(),
            dim=0,
        ).item()
        for layer in per_layer
    }
    orientation_score = sum(raw_axis_cosines.values()) / len(raw_axis_cosines)
    if orientation_score == 0:
        raise ValueError(
            "vjp_delta has exactly zero chosen/rejected orientation score"
        )
    orientation_flipped = orientation_score < 0
    oriented = (
        {layer: -vector for layer, vector in per_layer.items()}
        if orientation_flipped
        else per_layer
    )
    return oriented, raw_axis_cosines, orientation_score, orientation_flipped


def _valid_mask(attention_mask: Tensor, skip_first: int) -> Tensor:
    real_len = attention_mask.sum(dim=1, keepdim=True)
    position = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    mask = (position[None, :] >= skip_first) & (position[None, :] < real_len - 1)
    return mask & attention_mask.bool()


@torch.no_grad()
def _target_class_mean(
    model: nn.Module,
    tok,
    prompts: list[str],
    target_layer: int,
    *,
    batch_size: int,
    max_length: int,
    label: str,
) -> Float[Tensor, "d"]:
    blocks = _get_blocks(model)
    device = next(model.parameters()).device
    total = None
    count = 0
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"vjp target {label}",
        mininterval=120,
        maxinterval=120,
    ):
        batch = prompts[start : start + batch_size]
        encoded = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)
        with _record_activations(blocks, (target_layer,)) as activations:
            model(**encoded)
        hidden = activations[target_layer]
        last_position = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(hidden.shape[0], device=device)
        last_hidden = hidden[batch_index, last_position].float()
        total = last_hidden.sum(0) if total is None else total + last_hidden.sum(0)
        count += last_hidden.shape[0]
    return total / count


def _pullback_class_mean(
    model: nn.Module,
    tok,
    prompts: list[str],
    source_layers: tuple[int, ...],
    target_layer: int,
    cotangent: Float[Tensor, "d"],
    *,
    batch_size: int,
    max_length: int,
    skip_first: int,
    label: str,
    cotangent_scope: str,
    source_scope: str,
) -> tuple[
    dict[int, Float[Tensor, "d"]],
    dict[int, Float[Tensor, "d"]],
    dict[int, Float[Tensor, "d"]],
    dict[int, Float[Tensor, "d"]],
]:
    if len(prompts) < 2:
        raise ValueError("vjp_delta split-half diagnostics need at least two prompts per class")
    blocks = _get_blocks(model)
    device = next(model.parameters()).device
    d_model = cotangent.shape[-1]
    sums = {
        layer: torch.zeros(d_model, dtype=torch.float32, device=device)
        for layer in source_layers
    }
    half_sums = [
        {
            layer: torch.zeros(d_model, dtype=torch.float32, device=device)
            for layer in source_layers
        }
        for _ in range(2)
    ]
    activation_sums = {
        layer: torch.zeros(d_model, dtype=torch.float32, device=device)
        for layer in source_layers
    }
    count = 0
    half_counts = [0, 0]
    graph_root = min(source_layers)

    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"vjp pullback {label}",
        mininterval=120,
        maxinterval=120,
    ):
        batch = prompts[start : start + batch_size]
        encoded = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            padding_side="right",
        ).to(device)
        valid = _valid_mask(encoded["attention_mask"], skip_first)
        if valid.sum(dim=1).min() == 0:
            raise ValueError(
                f"a {label} prompt has no valid positions for skip_first={skip_first}"
            )

        recorded_layers = (*source_layers, target_layer)
        with (
            _record_activations(blocks, recorded_layers, graph_root=graph_root) as activations,
            torch.enable_grad(),
        ):
            model(**encoded)
            target_hidden = activations[target_layer]
            source_hidden = [activations[layer] for layer in source_layers]
            if cotangent_scope == "all_valid":
                target_mask = valid
            elif cotangent_scope == "last_token":
                last_position = encoded["attention_mask"].sum(dim=1) - 1
                target_mask = torch.zeros_like(valid)
                target_mask[
                    torch.arange(target_mask.shape[0], device=device), last_position
                ] = True
            else:
                raise ValueError(f"unknown cotangent_scope={cotangent_scope!r}")
            grad_output = cotangent.detach().to(target_hidden).view(1, 1, d_model)
            grad_output = grad_output * target_mask.unsqueeze(-1)
            gradients = torch.autograd.grad(
                target_hidden,
                source_hidden,
                grad_outputs=grad_output,
            )

        if source_scope not in ("all_valid", "last_token"):
            raise ValueError(f"unknown source_scope={source_scope!r}")
        parity = torch.arange(start, start + len(batch), device=device) % 2
        valid_count = valid.sum(dim=1, keepdim=True).float()
        last_position = encoded["attention_mask"].sum(dim=1) - 1
        batch_index = torch.arange(len(batch), device=device)

        for layer, gradient, hidden in zip(source_layers, gradients, source_hidden):
            if source_scope == "all_valid":
                per_prompt = (
                    gradient.float() * valid.unsqueeze(-1)
                ).sum(dim=1) / valid_count
            else:
                per_prompt = gradient[batch_index, last_position].float()
            sums[layer] += per_prompt.sum(0)
            valid_length = valid.sum(dim=1)
            for quantile in (0.25, 0.5, 0.75, 1.0):
                sample_position = skip_first + (
                    (valid_length - 1).clamp(min=0).float() * quantile
                ).long()
                activation_sums[layer] += (
                    hidden.detach()[batch_index, sample_position].float().sum(0)
                )
            for half in (0, 1):
                half_sums[half][layer] += per_prompt[parity == half].sum(0)

        count += len(batch)
        for half in (0, 1):
            half_counts[half] += int((parity == half).sum())

    return (
        {layer: sums[layer] / count for layer in source_layers},
        {layer: half_sums[0][layer] / half_counts[0] for layer in source_layers},
        {layer: half_sums[1][layer] / half_counts[1] for layer in source_layers},
        {layer: activation_sums[layer] / (4 * count) for layer in source_layers},
    )


@register
class VjpDelta:
    name = "vjp_delta"
    extract_from_prompts = True

    @staticmethod
    def extract(
        model: nn.Module,
        tok,
        pos_prompts: list[str],
        neg_prompts: list[str],
        cfg: VjpDeltaC,
        *,
        batch_size: int,
        max_length: int,
    ) -> dict[int, dict[str, dict[str, Tensor]]]:
        model.requires_grad_(False)
        blocks = _get_blocks(model)
        source_layers = cfg.layers
        if source_layers is None or not source_layers:
            raise ValueError("vjp_delta requires explicit source layers")

        n_blocks = len(blocks)
        target_layer = cfg.target_layer
        if target_layer is None:
            target_layer = n_blocks - DEFAULT_TARGET_LAYER_FROM_END
        elif target_layer < 0:
            target_layer = n_blocks + target_layer
        if not 0 <= target_layer < n_blocks:
            raise ValueError(
                f"vjp_delta target layer {target_layer} is outside [0, {n_blocks})"
            )
        if max(source_layers) >= target_layer:
            raise ValueError(
                f"vjp_delta source layers {source_layers} must precede target layer "
                f"{target_layer}"
            )

        h_pos = _target_class_mean(
            model,
            tok,
            pos_prompts,
            target_layer,
            batch_size=batch_size,
            max_length=max_length,
            label="pos",
        )
        h_neg = _target_class_mean(
            model,
            tok,
            neg_prompts,
            target_layer,
            batch_size=batch_size,
            max_length=max_length,
            label="neg",
        )
        cotangent = h_pos - h_neg
        dimension_energy = cotangent.float().square()
        top_dimension_count = min(5, dimension_energy.numel())
        top_share = (
            dimension_energy.topk(top_dimension_count).values.sum()
            / (dimension_energy.sum() + ε)
        ).item()
        logger.info(
            f"vjp_delta target={target_layer} |pos|={h_pos.norm():.3f} "
            f"|neg|={h_neg.norm():.3f} |cotangent|={cotangent.norm():.3f} "
            f"top-{top_dimension_count} energy share={top_share:.3f}; "
            "SHOULD: nonzero cotangent without a few dimensions dominating"
        )

        pos, pos_half_0, pos_half_1, pos_acts = _pullback_class_mean(
            model,
            tok,
            pos_prompts,
            source_layers,
            target_layer,
            cotangent,
            batch_size=batch_size,
            max_length=max_length,
            skip_first=cfg.skip_first,
            label="pos",
            cotangent_scope=cfg.cotangent_scope,
            source_scope=cfg.source_scope,
        )
        neg, neg_half_0, neg_half_1, neg_acts = _pullback_class_mean(
            model,
            tok,
            neg_prompts,
            source_layers,
            target_layer,
            cotangent,
            batch_size=batch_size,
            max_length=max_length,
            skip_first=cfg.skip_first,
            label="neg",
            cotangent_scope=cfg.cotangent_scope,
            source_scope=cfg.source_scope,
        )

        delta = {layer: pos[layer] - neg[layer] for layer in source_layers}
        half_0 = {
            layer: pos_half_0[layer] - neg_half_0[layer] for layer in source_layers
        }
        half_1 = {
            layer: pos_half_1[layer] - neg_half_1[layer] for layer in source_layers
        }
        mean = {
            layer: 0.5 * (pos[layer] + neg[layer]) for layer in source_layers
        }
        activation_axis = {
            layer: pos_acts[layer] - neg_acts[layer] for layer in source_layers
        }
        delta, raw_axis_cosines, orientation_score, orientation_flipped = (
            orient_vjp_delta(delta, activation_axis)
        )
        logger.info(
            "vjp_delta chosen/rejected orientation: "
            f"mean raw cos={orientation_score:+.3f} "
            f"{'(GLOBAL FLIP)' if orientation_flipped else '(kept)'}; "
            + " ".join(
                f"l{layer}={raw_axis_cosines[layer]:+.2f}"
                for layer in source_layers
            )
            + "\nSHOULD: mean raw cos far from 0 gives a stable global sign; near 0 "
              "means the VJP separation gradient is nearly orthogonal to the "
              "activation contrast"
        )
        rows = []
        reliabilities = []
        for layer in source_layers:
            reliability = F.cosine_similarity(
                half_0[layer], half_1[layer], dim=0
            ).item()
            reliabilities.append(reliability)
            axis_cosine = F.cosine_similarity(
                delta[layer], activation_axis[layer], dim=0
            ).item()
            delta_norm = delta[layer].norm()
            mean_norm = mean[layer].norm()
            rows.append([
                layer,
                delta_norm,
                mean_norm,
                delta_norm / (mean_norm + ε),
                reliability,
                axis_cosine,
            ])
        logger.info(
            "vjp_delta pullback diagnostics\n"
            + tabulate(
                rows,
                headers=[
                    "layer",
                    "|delta|",
                    "|mean|",
                    "|d|/|m|",
                    "split_half_cos",
                    "axis_cos",
                ],
                tablefmt="tsv",
                floatfmt=".3f",
            )
            + f"\nmean split_half_cos={sum(reliabilities) / len(reliabilities):+.3f}"
            "\nTODO validate: positive split-half cosine means the class contrast is "
            "reproducible; a tiny |delta|/|mean| means common-mode pullback dominates; "
            "axis_cos near zero means the VJP points away from the source-layer persona axis."
        )

        return {
            layer: {
                "shared": {},
                "stacked": {
                    "v": (
                        _unit_vector(delta[layer]) if cfg.normalize else delta[layer]
                    ).unsqueeze(0)
                },
            }
            for layer in source_layers
        }

    @staticmethod
    def apply(
        _mod,
        _x: Float[Tensor, "b s d"],
        y: Float[Tensor, "b s d"],
        _shared: dict[str, Tensor],
        stacked: dict[str, Tensor],
        cfg: VjpDeltaC,
    ) -> Float[Tensor, "b s d"]:
        vector = stacked["v"].to(y).sum(dim=0)
        if cfg.apply_mode == "add":
            return y + cfg.coeff * vector
        if cfg.apply_mode == "damp_amp":
            unit = vector / (vector.norm() + ε)
            projection = (y * unit).sum(dim=-1, keepdim=True)
            return y + (math.exp(cfg.coeff) - 1.0) * projection * unit
        raise ValueError(f"unknown vjp_delta apply_mode={cfg.apply_mode!r}")
