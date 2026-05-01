"""AIRiskDilemmas benchmark with symmetric per-value-class logratios in nats.

Unlike the paper's Elo analysis, this benchmark keeps the core action-choice
prompt (`Action 1` vs `Action 2`) but reports signed logratios for each value
class:

    lr_action = log P(Action 1) - log P(Action 2)
    lr_value  = label_value * lr_action

where `label_value = +1` if the value appears only on Action 1, `-1` if only on
Action 2. This makes value preference shifts directly comparable to the
daily-dilemmas benchmark.
"""
from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from loguru import logger

import steering_lite as sl
from steering_lite.eval.airisk_dilemmas import (
    AiriskEvalRow,
    all_value_classes,
    format_training_prompt,
    get_choice_ids,
    load_eval_rows,
    load_pairs,
    score_mcq,
    score_mcq_guided,
)

@dataclass
class BenchmarkConfig:
    model: str = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    method: str = "mean_diff"
    target: str = "Truthfulness"
    layers: tuple[int, ...] = ()
    coeff: float = 2.0
    n_train: int = 32
    n_eval: int | None = None
    max_seq_length: int = 512
    torch_dtype: str = "float32"
    device: str = "cpu"
    seed: int = 0
    output_dir: Path = Path("outputs/airisk_dilemmas")
    guided: bool = True
    max_think_tokens: int = 128


def cfg_for_method(bcfg: BenchmarkConfig, dtype: torch.dtype):
    common = dict(layers=bcfg.layers, coeff=bcfg.coeff, dtype=dtype, seed=bcfg.seed)
    table = {
        "mean_diff": sl.MeanDiffC(**common),
        "mean_centred": sl.MeanDiffC(**common, subtract_corpus_mean=True),
        "pca": sl.PCAC(**common),
        "topk_clusters": sl.TopKClustersC(**common, k=min(bcfg.n_train, 4)),
        "cosine_gated": sl.CosineGatedC(**common, tau=0.0),
        "sspace": sl.SSpaceC(**common, r=min(bcfg.n_train, 4)),
        "spherical": sl.SphericalC(**common),
        "directional_ablation": sl.DirectionalAblationC(**common),
        "chars": sl.CHaRSC(**common, k=min(bcfg.n_train, 4)),
        "linear_act": sl.LinearAcTC(**common),
        "angular_steering": sl.AngularSteeringC(**common),
    }
    if bcfg.method == "baseline":
        return None
    return table[bcfg.method]


def _parse_layers(s: str) -> tuple[int, ...]:
    s = s.strip().lower()
    if s in ("mid", "all"):
        return (-1,) if s == "mid" else (-2,)
    return tuple(int(x) for x in s.split(","))


def _resolve_layers(layers: tuple[int, ...], n_hidden: int) -> tuple[int, ...]:
    if layers == (-1,):
        lo, hi = int(n_hidden * 0.30), int(n_hidden * 0.80)
        return tuple(range(lo, hi))
    if layers == (-2,):
        return tuple(range(n_hidden))
    return layers


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _mean_int(xs: list[int]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _real_train_and_rows(tok, target: str, n_train: int, n_eval_rows: int | None, seed: int):
    pairs = load_pairs(target, seed=seed)
    if len(pairs) < n_train:
        valid = ", ".join(all_value_classes())
        raise RuntimeError(
            f"only {len(pairs)} train pairs for value_class={target!r}; need n_train={n_train}. "
            f"Valid value classes: {valid}"
        )
    train_pairs = pairs[:n_train]
    pos = [format_training_prompt(p.dilemma, p.action_1, p.action_2, "1", tok) for p in train_pairs if target in p.values_action_1]
    pos += [format_training_prompt(p.dilemma, p.action_1, p.action_2, "2", tok) for p in train_pairs if target in p.values_action_2]
    neg = [format_training_prompt(p.dilemma, p.action_1, p.action_2, "2", tok) for p in train_pairs if target in p.values_action_1]
    neg += [format_training_prompt(p.dilemma, p.action_1, p.action_2, "1", tok) for p in train_pairs if target in p.values_action_2]
    if len(pos) != len(neg):
        raise RuntimeError(f"train prompt mismatch pos={len(pos)} neg={len(neg)} for target={target!r}")
    eval_rows = load_eval_rows(seed=seed, max_rows=n_eval_rows)
    return pos, neg, eval_rows


def _eval_run(
    model, tok, eval_rows: list[AiriskEvalRow], choice_ids, device,
    *, guided: bool = False, max_think_tokens: int = 128,
) -> list[dict]:
    per_row = []
    n_nan = 0
    for r in eval_rows:
        if guided:
            out = score_mcq_guided(
                model, tok, r.dilemma, r.action_1, r.action_2, choice_ids, device,
                max_think_tokens=max_think_tokens,
            )
        else:
            out = score_mcq(model, tok, r.dilemma, r.action_1, r.action_2, choice_ids, device)
        if out["logratio"] != out["logratio"]:
            n_nan += 1
        per_row.append({
            "dilemma_idx": r.dilemma_idx,
            "logratio": out["logratio"],
            "pmass": out["pmass"],
            "max_p": out["max_p"],
            "logp": out["logp"],
            "think_tokens": out["think_tokens"],
            "value_labels": r.value_labels,
        })
    if n_nan > 0:
        logger.warning(f"NaN logratio on {n_nan}/{len(eval_rows)} AIRisk rows")
    return per_row


def _aggregate_effects(base_rows: list[dict], steered_rows: list[dict], value_classes: list[str]) -> dict[str, dict]:
    effects: dict[str, dict] = {}
    for value_class in value_classes:
        paired_base: list[float] = []
        paired_steered: list[float] = []
        for br, sr in zip(base_rows, steered_rows):
            label = br["value_labels"].get(value_class)
            if label is None:
                continue
            if br["logratio"] != br["logratio"] or sr["logratio"] != sr["logratio"]:
                continue
            paired_base.append(float(label) * float(br["logratio"]))
            paired_steered.append(float(label) * float(sr["logratio"]))
        n = len(paired_base)
        if n == 0:
            continue
        base_mean = _mean(paired_base)
        steered_mean = _mean(paired_steered)
        effects[value_class] = {
            "base": base_mean,
            "steered": steered_mean,
            "delta": steered_mean - base_mean,
            "n": n,
        }
    return effects


def run(cfg: BenchmarkConfig) -> dict:
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, cfg.torch_dtype)
    logger.info(f"loading model={cfg.model} dtype={dtype} device={cfg.device}")
    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token or "<pad>"
    model = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=dtype).to(cfg.device).eval()

    cfg_obj = model.config
    n_hidden = getattr(cfg_obj, "num_hidden_layers", None)
    if n_hidden is None and hasattr(cfg_obj, "text_config"):
        n_hidden = cfg_obj.text_config.num_hidden_layers
    if n_hidden is None:
        raise AttributeError(f"can't find num_hidden_layers on {type(cfg_obj).__name__}")
    cfg.layers = _resolve_layers(cfg.layers, n_hidden)
    logger.info(f"layers resolved: {cfg.layers} (model has {n_hidden} hidden layers)")
    value_classes = all_value_classes()

    pos, neg, eval_rows = _real_train_and_rows(tok, cfg.target, cfg.n_train, cfg.n_eval, cfg.seed)
    choice_ids = get_choice_ids(tok)
    logger.info(
        f"target={cfg.target} n_train={len(pos)} n_eval_rows={len(eval_rows)} "
        f"action1_ids={len(choice_ids[1])} action2_ids={len(choice_ids[0])}"
    )

    use_guided = cfg.guided
    logger.info(f"eval mode: {'guided CoT' if use_guided else 'teacher-forced'} "
                f"(max_think_tokens={cfg.max_think_tokens if use_guided else 0})")
    base_rows = _eval_run(
        model, tok, eval_rows, choice_ids, cfg.device,
        guided=use_guided, max_think_tokens=cfg.max_think_tokens,
    )

    s_cfg = cfg_for_method(cfg, dtype)
    vector_norms = {}
    steered_rows = base_rows
    if s_cfg is not None:
        logger.info(f"extracting steering vectors via method={cfg.method}")
        vectors = sl.train(model, tok, pos, neg, s_cfg, batch_size=4, max_length=cfg.max_seq_length)
        vector_norms = {li: {k: float(v.float().norm()) for k, v in d.items()} for li, d in vectors.state.items()}
        sl.attach(model, s_cfg, vectors)
        steered_rows = _eval_run(
            model, tok, eval_rows, choice_ids, cfg.device,
            guided=use_guided, max_think_tokens=cfg.max_think_tokens,
        )
        sl.detach(model)

    effects = _aggregate_effects(base_rows, steered_rows, value_classes)
    if cfg.target not in effects:
        raise RuntimeError(f"target {cfg.target!r} not present in AIRisk effects")

    base_valid = sum(1 for r in base_rows if r["logratio"] == r["logratio"])
    steered_valid = sum(1 for r in steered_rows if r["logratio"] == r["logratio"])
    base_pmass = _mean([r["pmass"] for r in base_rows])
    steered_pmass = _mean([r["pmass"] for r in steered_rows])
    base_think_tokens = [int(r["think_tokens"]) for r in base_rows]
    steered_think_tokens = [int(r["think_tokens"]) for r in steered_rows]

    result = {
        "config": {**asdict(cfg), "output_dir": str(cfg.output_dir)},
        "effects": effects,
        "summary": {
            "target": cfg.target,
            "target_effect": effects[cfg.target]["delta"],
            "n_eval_rows": len(eval_rows),
            "n_valid_base": base_valid,
            "n_valid_steered": steered_valid,
            "base_pct_valid": base_valid / max(len(base_rows), 1),
            "steered_pct_valid": steered_valid / max(len(steered_rows), 1),
            "base_pmass_mean": base_pmass,
            "steered_pmass_mean": steered_pmass,
            "base_think_tokens_mean": _mean_int(base_think_tokens),
            "steered_think_tokens_mean": _mean_int(steered_think_tokens),
        },
        "vector_norms": vector_norms,
    }

    os.makedirs(cfg.output_dir, exist_ok=True)
    out_stem = (
        f"{cfg.model.replace('/', '--')}__{cfg.method}__{cfg.target.replace(' ', '_')}"
        f"__c{cfg.coeff}__seed{cfg.seed}__{int(time.time())}"
    )
    out_path = cfg.output_dir / f"{out_stem}.json"
    out_path.write_text(json.dumps(result, indent=2))

    csv_path = cfg.output_dir / f"{out_stem}__per_row.csv"
    fieldnames = [
        "idx", "dilemma_idx", "method", "coeff", "model", "target", "seed",
        "logratio_action", "pmass", "max_p", "think_tokens",
        f"logratio_{cfg.target}",
    ] + [f"value_{v}" for v in value_classes]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in steered_rows:
            target_label = r["value_labels"].get(cfg.target)
            row = {
                "idx": str(r["dilemma_idx"]),
                "dilemma_idx": r["dilemma_idx"],
                "method": cfg.method,
                "coeff": float(cfg.coeff) if s_cfg is not None else 0.0,
                "model": cfg.model,
                "target": cfg.target,
                "seed": cfg.seed,
                "logratio_action": r["logratio"],
                "pmass": r["pmass"],
                "max_p": r["max_p"],
                "think_tokens": int(r["think_tokens"]),
                f"logratio_{cfg.target}": "" if target_label is None else target_label * r["logratio"],
            }
            for value_class in value_classes:
                label = r["value_labels"].get(value_class)
                row[f"value_{value_class}"] = "" if label is None else label * r["logratio"]
            writer.writerow(row)
    logger.info(f"wrote {out_path} and {csv_path}")
    logger.info(
        f"target_effect={effects[cfg.target]['delta']:+.4f} | "
        f"valid={base_valid}/{steered_valid}/{len(base_rows)} "
        f"pmass={base_pmass:.3f}/{steered_pmass:.3f} "
        f"think={_mean_int(base_think_tokens):.1f}/{_mean_int(steered_think_tokens):.1f}"
    )
    return result


def _parse_args():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BenchmarkConfig.model)
    p.add_argument("--method", default=BenchmarkConfig.method)
    p.add_argument("--target", default=BenchmarkConfig.target,
                   help="AIRisk value class to steer toward, e.g. Truthfulness, Protection")
    p.add_argument("--layers", default="mid")
    p.add_argument("--coeff", type=float, default=BenchmarkConfig.coeff)
    p.add_argument("--n-train", type=int, default=BenchmarkConfig.n_train)
    p.add_argument("--n-eval", type=int, default=BenchmarkConfig.n_eval)
    p.add_argument("--torch-dtype", default=BenchmarkConfig.torch_dtype)
    p.add_argument("--device", default=BenchmarkConfig.device)
    p.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    p.add_argument("--output-dir", default=str(BenchmarkConfig.output_dir))
    p.add_argument("--no-guided", action="store_true")
    p.add_argument("--max-think-tokens", type=int, default=BenchmarkConfig.max_think_tokens)
    args = p.parse_args()
    return BenchmarkConfig(
        model=args.model,
        method=args.method,
        target=args.target,
        layers=_parse_layers(args.layers),
        coeff=args.coeff,
        n_train=args.n_train,
        n_eval=args.n_eval,
        torch_dtype=args.torch_dtype,
        device=args.device,
        seed=args.seed,
        output_dir=Path(args.output_dir),
        guided=not args.no_guided,
        max_think_tokens=args.max_think_tokens,
    )


if __name__ == "__main__":
    run(_parse_args())
