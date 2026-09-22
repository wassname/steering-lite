"""Cached local BS-bench stages, adapted from vjp-steering's walk path."""
from __future__ import annotations

from pathlib import Path

import torch

import steering_lite as sl
from .cache import cached_stage
from .generation import generation_inputs, generate, read_dev_cohort, vector_hash

METHODS = (
    "bare", "prompting", "random", "mean_diff", "pca", "kv_cache_gram",
    "vjp_delta", "vjp_cache",
)


def method_config(method: str, *, layers: tuple[int, ...], target_layer: int, seed: int = 0):
    common = {"layers": layers, "coeff": 0.2, "dtype": torch.float32, "seed": seed}
    configs = {
        "random": sl.RandomC(**common),
        "mean_diff": sl.MeanDiffC(**common),
        "pca": sl.PCAC(**common),
        "kv_cache_gram": sl.KVCacheGramC(**common, r=2),
        "vjp_delta": sl.VjpDeltaC(**common, target_layer=target_layer, skip_first=16),
        "vjp_cache": sl.VjpCacheC(**common, target_layer=target_layer, skip_first=16),
    }
    if method not in configs:
        raise ValueError(f"{method!r} has no steering config")
    return configs[method]


def _vector_answers(model, tokenizer, prompts: list[str], vector, *, batch_size: int, max_new_tokens: int) -> list[str]:
    with vector(model):
        return generate(model, tokenizer, prompts, batch_size, max_new_tokens)


def run_method(
    model,
    tokenizer,
    method: str,
    pos_prompts: list[str],
    neg_prompts: list[str],
    *,
    vector_dir: Path,
    rows: list[dict] | None = None,
    layers: tuple[int, ...],
    target_layer: int,
    batch_size: int = 1,
    max_length: int = 64,
    max_new_tokens: int = 2,
) -> dict:
    """Run one bare/prompt/vector method through the normal local pipeline."""
    if method not in METHODS:
        raise ValueError(f"unknown benchmark method {method!r}")
    rows = read_dev_cohort() if rows is None else rows
    prompts = generation_inputs(tokenizer, rows)
    if method == "bare":
        answers = generate(model, tokenizer, prompts, batch_size, max_new_tokens)
        return {"method": method, "answers": answers}
    if method == "prompting":
        answers = generate(model, tokenizer, generation_inputs(tokenizer, rows, persona="sycophantic"), batch_size, max_new_tokens)
        return {"method": method, "answers": answers}

    cfg = method_config(method, layers=layers, target_layer=target_layer)
    vector = sl.train(model, tokenizer, pos_prompts, neg_prompts, cfg, batch_size=batch_size, max_length=max_length)
    vector_dir.mkdir(parents=True, exist_ok=True)
    path = vector_dir / f"{method}.safetensors"
    vector.save(str(path))
    restored = sl.Vector.load(str(path))
    answers = _vector_answers(model, tokenizer, prompts, restored, batch_size=batch_size, max_new_tokens=max_new_tokens)
    return {
        "method": method,
        "answers": answers,
        "vector": path.name,
        "vector_sha256": vector_hash(restored),
        "config": restored.cfg.to_dict(),
    }


def cached_method(
    root: Path,
    *,
    model_identity: dict,
    data_identity: dict,
    method: str,
    config: dict,
    prompts: list[str],
    compute,
) -> dict:
    """Cache a serialisable completed stage with all benchmark inputs keyed."""
    return cached_stage(
        root,
        "generation",
        model=model_identity,
        data=data_identity,
        method=method,
        config=config,
        prompts=prompts,
        compute=compute,
    )
