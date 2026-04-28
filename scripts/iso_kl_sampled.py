"""Iso-KL calibration: find per-method coeff so KL(steer || base) stays bounded
over the first T_CALIB generated tokens.

Two modes:
  --mode greedy  : do_sample=False, target = max_t KL_sb over first T tokens (deterministic)
  --mode sampled : do_sample=True,  target = mean_t KL_sb (noisy, on-policy)

Recommended: greedy mode with --target-kl 1.0 (no token spikes above 1 nat).
Greedy is deterministic so bisection converges fast and reproducibly.
At the chosen coeff we ALSO run a sampled validation pass reporting
sampled p50 / p90 / max so the user sees what deployment will look like.

Newton-ish bisection: standard interval bisection with an exponential bracket.
Newton-Raphson would need d(stat)/d(coeff) which we don't have analytically;
bisection with the deterministic greedy stat converges in ~6-8 iters to tol=0.05.
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
from steering_lite.daily_dilemmas import load_pairs, make_prompt


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


def build_prompts(tok, situations, chat: bool, enable_thinking: bool):
    out = []
    for s in situations:
        if chat:
            messages = [{"role": "user", "content": s}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=enable_thinking)
        else:
            text = make_prompt(s)
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


@torch.no_grad()
def measure_kl(model, prompts, T, vectors, s_cfg, eos_id, pad_id,
               do_sample, temperature, top_p, top_k, device):
    """For each prompt: with steering attached, generate T tokens; then score under
    base (detached) and steer (re-attached). Compute per-position KL_sb.
    Returns mean / p50 / p90 / max plus per-token-index aggregates."""
    all_kls = []
    per_t = [[] for _ in range(T)]
    for pids in prompts:
        sl.attach(model, s_cfg, vectors)
        try:
            gen = generate_one(model, pids, T, eos_id, pad_id, do_sample,
                               temperature, top_p, top_k, device)
        finally:
            sl.detach(model)
        n_gen = gen.shape[0]
        if n_gen == 0:
            continue
        full = torch.cat([pids.to(device), gen]).unsqueeze(0)
        n_p = pids.shape[0]
        logits_base = model(full).logits.float()
        logp_base = torch.log_softmax(logits_base, dim=-1)[0]
        sl.attach(model, s_cfg, vectors)
        try:
            logits_steer = model(full).logits.float()
        finally:
            sl.detach(model)
        logp_steer = torch.log_softmax(logits_steer, dim=-1)[0]
        slc = slice(n_p - 1, n_p - 1 + n_gen)
        kls = kl_at_positions(logp_steer[slc], logp_base[slc]).cpu()
        all_kls.append(kls)
        for i in range(n_gen):
            per_t[i].append(float(kls[i]))
    cat = torch.cat(all_kls)
    return {
        "kl_mean": float(cat.mean()),
        "kl_p50": float(cat.quantile(0.50)),
        "kl_p90": float(cat.quantile(0.90)),
        "kl_p95": float(cat.quantile(0.95)),
        "kl_max": float(cat.max()),
        "n_pos": int(cat.numel()),
        "per_t_mean": [sum(xs) / len(xs) if xs else 0.0 for xs in per_t],
        "per_t_max": [max(xs) if xs else 0.0 for xs in per_t],
    }


def calibrate(model, method, layers, dtype, seed, n_train,
              vectors, prompts, T, target_kl, target_stat,
              eos_id, pad_id, do_sample, temperature, top_p, top_k,
              device, tol=0.05, max_iters=12):
    is_spherical = method == "spherical"
    if is_spherical:
        lo, hi = 0.001, 0.5
    else:
        lo, hi = 0.05, 16.0

    history = []

    def eval_at(c):
        s_cfg = make_cfg(method, layers, c, dtype, seed, n_train)
        m = measure_kl(model, prompts, T, vectors, s_cfg, eos_id, pad_id,
                       do_sample, temperature, top_p, top_k, device)
        history.append({"coeff": c, **{k: v for k, v in m.items() if k not in ("per_t_mean", "per_t_max")}})
        logger.info(f"  [{method}] c={c:.4f} mean={m['kl_mean']:.3f} "
                    f"p50={m['kl_p50']:.3f} p90={m['kl_p90']:.3f} p95={m['kl_p95']:.3f} max={m['kl_max']:.3f} n={m['n_pos']}")
        return m[target_stat]

    mid = (lo * hi) ** 0.5
    v_mid = eval_at(mid)
    if v_mid < target_kl:
        c = mid
        while c < hi:
            c *= 2.0
            v = eval_at(c)
            if v >= target_kl:
                lo, hi = c / 2.0, c
                break
        else:
            return c, history
    else:
        c = mid
        while c > lo:
            c /= 2.0
            v = eval_at(c)
            if v <= target_kl:
                lo, hi = c, c * 2.0
                break
        else:
            return c, history

    for _ in range(max_iters):
        m = (lo + hi) / 2.0
        v = eval_at(m)
        if abs(v - target_kl) < tol:
            return m, history
        if v < target_kl:
            lo = m
        else:
            hi = m

    return (lo + hi) / 2.0, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target", default="honesty")
    p.add_argument("--layers", default="4")
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

    layers = tuple(int(x) for x in args.layers.split(","))
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

    do_sample_calib = args.mode == "sampled"
    all_summary = []
    for seed in seeds:
        logger.info(f"########## seed={seed} mode={args.mode} stat={args.target_stat} target_kl={args.target_kl} ##########")
        random.seed(seed); torch.manual_seed(seed)

        pairs = load_pairs(args.target, seed=seed)
        train_pairs = pairs[: args.n_train]
        calib_pairs = pairs[args.n_train : args.n_train + args.n_calib]
        val_pairs = pairs[args.n_train + args.n_calib : args.n_train + args.n_calib + args.n_validate]
        pos = [make_prompt(pp.situation) + pp.action_pos for pp in train_pairs]
        neg = [make_prompt(pp.situation) + pp.action_neg for pp in train_pairs]
        logger.info(f"target={args.target} n_train={len(pos)} n_calib={len(calib_pairs)} n_val={len(val_pairs)}")

        prompts_calib = build_prompts(tok, [pp.situation for pp in calib_pairs],
                                      args.chat_template, args.enable_thinking)
        prompts_val = build_prompts(tok, [pp.situation for pp in val_pairs],
                                    args.chat_template, args.enable_thinking)

        for method in methods:
            logger.info(f"=== seed={seed} {method} ===")
            s_cfg0 = make_cfg(method, layers, 1.0, dtype, seed, args.n_train)
            vectors = sl.train(model, tok, pos, neg, s_cfg0, batch_size=4, max_length=256)

            torch.manual_seed(seed)
            coeff_star, history = calibrate(
                model, method, layers, dtype, seed, args.n_train,
                vectors, prompts_calib, args.t_calib, args.target_kl, args.target_stat,
                eos_id, pad_id, do_sample_calib,
                args.temperature, args.top_p, args.top_k, args.device,
            )

            best = min(history, key=lambda h: abs(h[args.target_stat] - args.target_kl))

            logger.info(f"--- sampled validation [{method}] at coeff*={best['coeff']:.4f} ---")
            torch.manual_seed(seed)
            val_cfg = make_cfg(method, layers, best["coeff"], dtype, seed, args.n_train)
            val_m = measure_kl(model, prompts_val, args.t_calib, vectors, val_cfg, eos_id, pad_id,
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
