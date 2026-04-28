"""Iso-KL calibration runner.

Multi-method, multi-seed orchestration around `steering_lite.calibrate_iso_kl`.
The math (measurement + log-log secant solver) lives in the library; this
script handles config building, data loading, validation pass, and JSON output.
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path

import torch
from loguru import logger

import steering_lite as sl
from steering_lite.daily_dilemmas import load_pairs, format_mcq, format_mcq_thinking


METHODS = ["mean_diff", "pca", "topk_clusters", "cosine_gated", "sspace", "spherical"]


def make_cfg(method: str, layers: tuple[int, ...], coeff: float, dtype, seed: int, n_train: int):
    common = dict(layers=layers, coeff=coeff, dtype=dtype, seed=seed)
    table = {
        "mean_diff": sl.MeanDiffConfig(**common),
        "pca": sl.PCAConfig(**common),
        "topk_clusters": sl.TopKClustersConfig(**common, k=min(n_train, 4)),
        "cosine_gated": sl.CosineGatedConfig(**common, tau=0.0),
        "sspace": sl.SSpaceConfig(**common, r=min(n_train, 4)),
        "spherical": sl.SphericalConfig(**common),
    }
    return table[method]


def build_prompts(tok, pairs, chat: bool, enable_thinking: bool):
    """Build calib/val probe prompts. Uses MCQ-thinking format ending at
    `<think>\\n` so the generated tokens during calib ARE the thinking phase --
    same distribution that matters for guided CoT eval. Without this, calib
    would measure KL at the bare-prompt 'My choice:' position, but eval reads
    Yes/No at a position 32+ thinking tokens later under steering.
    """
    out = []
    for p in pairs:
        text = format_mcq_thinking(p.situation, p.action_pos, tok)
        ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids[0]
        out.append(ids)
    return out


@torch.no_grad()
def generate_one(model, prompt_ids, T, eos_id, pad_id,
                 do_sample, temperature, top_p, top_k, device):
    ids = prompt_ids.unsqueeze(0).to(device)
    kw = dict(max_new_tokens=T, pad_token_id=pad_id, eos_token_id=eos_id, num_return_sequences=1)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        kw.update(do_sample=False)
    out = model.generate(ids, **kw)
    return out[0, prompt_ids.shape[0]:]


@torch.no_grad()
def kl_at_positions(logp_steer, logp_base):
    p_s = logp_steer.exp()
    return (p_s * (logp_steer - logp_base)).sum(dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target", default="honesty")
    p.add_argument("--layers", default="mid",
                   help="comma-separated indices, OR 'mid' (30-80%% of model layers), OR 'all'.")
    p.add_argument("--mode", choices=["greedy", "sampled"], default="greedy")
    p.add_argument("--target-kl", type=float, default=1.0)
    p.add_argument("--target-stat", default=None)
    p.add_argument("--t-calib", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=4)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--seeds", default="0")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--n-validate", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/iso_kl")
    args = p.parse_args()

    if args.target_stat is None:
        # p95 is robust max over ~20 tokens; less noisy than true max but still bounds spikes
        args.target_stat = "kl_p95" if args.mode == "greedy" else "kl_mean"

    layers = None  # resolved after model load
    dtype = getattr(torch, args.torch_dtype)
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"loading model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token or "<pad>"
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    n_hidden = model.config.num_hidden_layers
    if args.layers.strip().lower() == "mid":
        layers = tuple(range(int(n_hidden * 0.30), int(n_hidden * 0.80)))
    elif args.layers.strip().lower() == "all":
        layers = tuple(range(n_hidden))
    else:
        layers = tuple(int(x) for x in args.layers.split(","))
    logger.info(f"layers resolved: {layers} (model has {n_hidden} hidden layers)")

    do_sample_calib = args.mode == "sampled"
    all_summary = []
    for seed in seeds:
        logger.info(f"########## seed={seed} mode={args.mode} stat={args.target_stat} target_kl={args.target_kl} ##########")
        random.seed(seed); torch.manual_seed(seed)

        pairs = load_pairs(args.target, seed=seed)
        train_pairs = pairs[: args.n_train]
        calib_pairs = pairs[args.n_train : args.n_train + args.n_calib]
        val_pairs = pairs[args.n_train + args.n_calib : args.n_train + args.n_calib + args.n_validate]
        pos = [format_mcq(pp.situation, pp.action_pos, tok) for pp in train_pairs]
        neg = [format_mcq(pp.situation, pp.action_neg, tok) for pp in train_pairs]
        logger.info(f"target={args.target} n_train={len(pos)} n_calib={len(calib_pairs)} n_val={len(val_pairs)}")

        prompts_calib = build_prompts(tok, calib_pairs, args.chat_template, args.enable_thinking)
        prompts_val = build_prompts(tok, val_pairs, args.chat_template, args.enable_thinking)

        for method in methods:
            logger.info(f"=== seed={seed} {method} ===")
            cfg0 = make_cfg(method, layers, 1.0, dtype, seed, args.n_train)
            vectors = sl.train(model, tok, pos, neg, cfg0, batch_size=4, max_length=256)

            torch.manual_seed(seed)
            bracket = (0.001, 0.5) if method == "spherical" else (0.05, 16.0)
            coeff_star, history = sl.calibrate_iso_kl(
                model, prompts_calib, cfg0, vectors,
                target_kl=args.target_kl, target_stat=args.target_stat,
                bracket=bracket, T=args.t_calib,
                eos_id=eos_id, pad_id=pad_id,
                do_sample=do_sample_calib,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                device=args.device,
            )

            best = min(history, key=lambda h: abs(h[args.target_stat] - args.target_kl))

            logger.info(f"--- sampled validation [{method}] at coeff*={best['coeff']:.4f} ---")
            torch.manual_seed(seed)
            from dataclasses import replace
            val_cfg = replace(cfg0, coeff=best["coeff"])
            val_m = sl.measure_kl(model, prompts_val, val_cfg, vectors,
                                   T=args.t_calib, eos_id=eos_id, pad_id=pad_id,
                                   do_sample=True,
                                   temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                                   device=args.device)
            logger.info(f"  [val sampled] mean={val_m['kl_mean']:.3f} p50={val_m['kl_p50']:.3f} "
                        f"p90={val_m['kl_p90']:.3f} max={val_m['kl_max']:.3f} n={val_m['n_pos']}")

            row = {
                "model": args.model, "method": method, "seed": seed,
                "calibrated_coeff": best["coeff"],
                "calib_mode": args.mode,
                "calib_target_stat": args.target_stat,
                "calib_target_kl": args.target_kl,
                "calib_kl_mean": best["kl_mean"],
                "calib_kl_p50": best["kl_p50"],
                "calib_kl_p90": best["kl_p90"],
                "calib_kl_p95": best["kl_p95"],
                "calib_kl_max": best["kl_max"],
                "val_sampled_kl_mean": val_m["kl_mean"],
                "val_sampled_kl_p50": val_m["kl_p50"],
                "val_sampled_kl_p90": val_m["kl_p90"],
                "val_sampled_kl_p95": val_m["kl_p95"],
                "val_sampled_kl_max": val_m["kl_max"],
                "val_n_pos": val_m["n_pos"],
                "val_per_t_mean": val_m["per_t_mean"],
                "val_per_t_max": val_m["per_t_max"],
                "iters": len(history),
                "history": history,
            }
            all_summary.append(row)

    out_path = out_dir / (
        f"iso_kl__{args.model.replace('/', '--')}__L{'_'.join(map(str, layers))}"
        f"__{args.mode}_{args.target_stat}_{args.target_kl}__T{args.t_calib}__N{args.n_calib}"
        f"__seeds{args.seeds.replace(',', '_')}__{int(time.time())}.json"
    )
    out_path.write_text(json.dumps({"args": vars(args), "summary": all_summary}, indent=2))
    logger.info(f"wrote {out_path}")
    print(f"\n# Calibration: mode={args.mode} target {args.target_stat}≈{args.target_kl} on first {args.t_calib} tokens, N_calib={args.n_calib}, N_val={args.n_validate}\n")
    print("| seed | method | coeff* | calib_p95 | calib_max | calib_mean | val_samp_mean | val_samp_p95 | val_samp_max | iters |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_summary:
        print(f"| {r['seed']} | {r['method']} | {r['calibrated_coeff']:.4f} | "
              f"{r['calib_kl_p95']:.3f} | {r['calib_kl_max']:.3f} | {r['calib_kl_mean']:.3f} | "
              f"{r['val_sampled_kl_mean']:.3f} | {r['val_sampled_kl_p95']:.3f} | {r['val_sampled_kl_max']:.3f} | "
              f"{r['iters']} |")


if __name__ == "__main__":
    main()
