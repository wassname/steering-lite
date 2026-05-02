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
from steering_lite._quiet import quiet_external_logs
from steering_lite.eval.airisk_dilemmas import load_pairs, format_training_prompt, format_mcq_thinking

quiet_external_logs()


METHODS = ["mean_diff", "pca", "topk_clusters", "cosine_gated", "sspace", "spherical"]


def make_cfg(method: str, layers: tuple[int, ...], coeff: float, dtype, seed: int, n_train: int):
    common = dict(layers=layers, coeff=coeff, dtype=dtype, seed=seed)
    table = {
        "mean_diff": sl.MeanDiffC(**common),
        "pca": sl.PCAC(**common),
        "topk_clusters": sl.TopKClustersC(**common, k=min(n_train, 4)),
        "cosine_gated": sl.CosineGatedC(**common, tau=0.0),
        "sspace": sl.SSpaceC(**common, r=min(n_train, 4)),
        "spherical": sl.SphericalC(**common),
    }
    return table[method]


def build_prompts(tok, pairs, target: str):
    """Build calib/val probe prompts ending at `<think>\\n`.

    Calibrates KL at the thinking-phase distribution, matching guided-CoT eval.
    """
    out = []
    for p in pairs:
        text = format_mcq_thinking(p.dilemma, p.action_1, p.action_2, tok)
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
    p.add_argument("--target", default="Truthfulness")
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
    p.add_argument("--n-validate", type=int, default=8)
    p.add_argument("--output-dir", default="outputs/iso_kl")
    p.add_argument("--separate-signs", action="store_true",
                   help="calibrate positive and negative coeffs separately")
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

    n_hidden = getattr(model.config, "num_hidden_layers", None)
    if n_hidden is None and hasattr(model.config, "text_config"):
        n_hidden = model.config.text_config.num_hidden_layers
    if n_hidden is None:
        raise AttributeError(f"can't find num_hidden_layers on {type(model.config).__name__}")
    if args.layers.strip().lower() == "mid":
        layers = tuple(range(int(n_hidden * 0.30), int(n_hidden * 0.80)))
    elif args.layers.strip().lower() == "all":
        layers = tuple(range(n_hidden))
    else:
        layers = tuple(int(x) for x in args.layers.split(","))
    logger.info(f"layers resolved: {layers} (model has {n_hidden} hidden layers)")

    do_sample_calib = args.mode == "sampled"
    all_summary = []
    all_summary_signed = []
    for seed in seeds:
        logger.info(f"########## seed={seed} mode={args.mode} stat={args.target_stat} target_kl={args.target_kl} ##########")
        random.seed(seed); torch.manual_seed(seed)

        pairs = load_pairs(args.target, seed=seed)
        train_pairs = pairs[: args.n_train]
        calib_pairs = pairs[args.n_train : args.n_train + args.n_calib]
        val_pairs = pairs[args.n_train + args.n_calib : args.n_train + args.n_calib + args.n_validate]
        pos = [format_training_prompt(pp.dilemma, pp.action_1, pp.action_2,
                                      "1" if args.target in pp.values_action_1 else "2", tok)
               for pp in train_pairs]
        neg = [format_training_prompt(pp.dilemma, pp.action_1, pp.action_2,
                                      "2" if args.target in pp.values_action_1 else "1", tok)
               for pp in train_pairs]
        logger.info(f"target={args.target} n_train={len(pos)} n_calib={len(calib_pairs)} n_val={len(val_pairs)}")

        prompts_calib = build_prompts(tok, calib_pairs, args.target)
        prompts_val = build_prompts(tok, val_pairs, args.target)

        for method in methods:
            logger.info(f"=== seed={seed} {method} ===")
            cfg0 = make_cfg(method, layers, 1.0, dtype, seed, args.n_train)
            v = sl.Vector.train(model, tok, pos, neg, cfg0, batch_size=4, max_length=256)
            signs = [("positive", +1.0), ("negative", -1.0)] if args.separate_signs else [("positive", +1.0)]
            signed_rows = []
            for direction, sign in signs:
                torch.manual_seed(seed)
                bracket = (0.001, 0.5) if method == "spherical" else (0.05, 16.0)
                coeff_star, history = sl.calibrate_iso_kl(
                    v, model, tok, prompts_calib,
                    target_kl=args.target_kl, target_stat=args.target_stat,
                    bracket=bracket, T=args.t_calib,
                    device=args.device, sign=sign,
                )

                best = min(history, key=lambda h: abs(h[args.target_stat] - args.target_kl))

                logger.info(f"--- sampled validation [{method} {direction}] at coeff*={best['coeff']:+.4f} ---")
                torch.manual_seed(seed)
                v.cfg.coeff = best["coeff"]
                val_m = sl.measure_kl(v, model, tok, prompts_val,
                                       T=args.t_calib, do_sample=True,
                                       device=args.device)
                logger.info(f"  [val sampled] mean={val_m['kl_mean']:.3f} p50={val_m['kl_p50']:.3f} "
                            f"p90={val_m['kl_p90']:.3f} max={val_m['kl_max']:.3f} n={val_m['n_pos']}")

                row = {
                    "model": args.model, "method": method, "seed": seed,
                    "direction": direction, "sign": sign,
                    "calibrated_coeff": best["coeff"],
                    "calibrated_coeff_abs": abs(best["coeff"]),
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
                signed_rows.append(row)
                all_summary_signed.append(row)

            # Backward-compatible one-row summary keeps the positive direction.
            all_summary.append(next(r for r in signed_rows if r["sign"] > 0))

    out_path = out_dir / (
        f"iso_kl__{args.model.replace('/', '--')}__L{'_'.join(map(str, layers))}"
        f"__{args.mode}_{args.target_stat}_{args.target_kl}__T{args.t_calib}__N{args.n_calib}"
        f"__seeds{args.seeds.replace(',', '_')}__{int(time.time())}.json"
    )
    out_path.write_text(json.dumps({
        "args": vars(args),
        "summary": all_summary,
        "summary_signed": all_summary_signed,
    }, indent=2))
    logger.info(f"wrote {out_path}")
    from tabulate import tabulate
    rows = [
        [r["seed"], r["method"], f"{r['calibrated_coeff']:.4f}",
         f"{r['calib_kl_p95']:.3f}", f"{r['calib_kl_max']:.3f}", f"{r['calib_kl_mean']:.3f}",
         f"{r['val_sampled_kl_mean']:.3f}", f"{r['val_sampled_kl_p95']:.3f}", f"{r['val_sampled_kl_max']:.3f}",
         r["iters"]]
        for r in all_summary
    ]
    logger.info(f"\n# Calibration: mode={args.mode} target {args.target_stat}≈{args.target_kl} T={args.t_calib} N_calib={args.n_calib} N_val={args.n_validate}\n"
                + tabulate(rows, headers=["seed", "method", "coeff*", "calib_p95", "calib_max", "calib_mean",
                                          "val_samp_mean", "val_samp_p95", "val_samp_max", "iters"],
                           tablefmt="tsv"))


if __name__ == "__main__":
    main()
