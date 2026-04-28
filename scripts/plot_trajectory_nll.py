"""Plot per-token NLL and flip rate over a long baseline trajectory.

Generates a sequence of 1024 tokens from the base model.
Computes per-token NLL of both steered and base models on this sequence.
Saves cumulative delta-NLL to CSV for plotting to check for compounding drift.

Usage:
    uv run python scripts/plot_trajectory_nll.py \
        --iso-tv-json outputs/iso_tv_n200/...json \
        --layers 4 --max-new 1024
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import json

import torch
import pandas as pd
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.daily_dilemmas import load_pairs, make_prompt


def make_cfg(method: str, layers, coeff: float, dtype, seed: int, n_train: int):
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


@torch.no_grad()
def trajectory_nll(model, tok, prompt: str, generated_ids, device) -> dict:
    """Return per-token NLL along the generated trajectory."""
    # input: prompt_ids + generated_ids
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)
    n_p = len(prompt_ids)
    
    # We pass the full past into the model to get logits for the generated tokens.
    # To get NLL of token i, we need log_{softmax}(logits[i-1]) from index i-1.
    full_ids = torch.cat([prompt_ids, generated_ids]).unsqueeze(0)
    
    logits = model(full_ids).logits.float()  # [1, seq_len, V]
    logp = torch.log_softmax(logits, dim=-1)
    
    # Target tokens are the generated ones: full_ids[0, n_p:]
    targets = full_ids[0, n_p:]
    
    # Logits predicting those targets come from [n_p-1 : end-1]
    pred_logp = logp[0, n_p-1 : full_ids.shape[1]-1]
    
    # Gather the logp of the actual target token at each step
    target_logp = pred_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
    # Did the argmax change?
    argmax_tokens = pred_logp.argmax(dim=-1)
    flips = (argmax_tokens != targets)
    
    return {
        "nll_per_token": -target_logp.cpu().numpy(),
        "flips_per_token": flips.cpu().numpy(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--target-value", default=None)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--situation-idx", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/trajectory")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/verbose.log", level="DEBUG", mode="a",
               format="{time:HH:mm:ss} | {level} | {message}")

    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = args.target_value or iso["args"].get("target", "harmless")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    if not rows:
        logger.error(f"no rows in {args.iso_tv_json} with seed={args.seed}")
        sys.exit(1)

    logger.info(f"model={model_id} layers={layers} seed={args.seed} max_new={args.max_new}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    import random
    random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = load_pairs(target_value, seed=args.seed)
    train_pairs = pairs[: args.n_train]
    pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
    neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
    
    eval_pool = pairs[args.n_train:]
    if not eval_pool: eval_pool = pairs
    p = eval_pool[args.situation_idx % len(eval_pool)]
    prompt = make_prompt(p.situation)
    logger.info(f"prompt: {p.situation[:80]!r}...")

    # 1. Base model greedy trajectory -- force full max_new tokens (override EOS)
    # so we can study long-trajectory behavior past natural stopping.
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    logger.info(f"generating base trajectory ({args.max_new} tokens, EOS suppressed)...")
    base_gen_ids = model.generate(
        ids, max_new_tokens=args.max_new, min_new_tokens=args.max_new,
        do_sample=False, temperature=1.0, pad_token_id=tok.pad_token_id,
        suppress_tokens=[tok.eos_token_id] if tok.eos_token_id is not None else None,
    )
    gen_ids = base_gen_ids[0, ids.shape[1]:]  # just the new tokens
    
    base_text = tok.decode(gen_ids, skip_special_tokens=False)
    logger.debug(f"\nBASE TRAJECTORY:\n{base_text}\n")
    
    base_res = trajectory_nll(model, tok, prompt, gen_ids, device)
    
    # SHOULD: Base nll per token should average around 0.5-2.0
    print(f"SHOULD: mean base NLL ~1.0; got {base_res['nll_per_token'].mean():.2f}")

    df_rows = []
    
    print("\n=== Trajectory Deltas (128 interval snapshots) ===")
    
    for r in rows:
        method = r["method"]
        coeff = r["calibrated_coeff"]
        
        cfg_train = make_cfg(method, layers, coeff, dtype, args.seed, args.n_train)
        vectors = sl.train(model, tok, pos, neg, cfg_train, batch_size=4, max_length=256)
        
        sl.attach(model, cfg_train, vectors)
        st_res = trajectory_nll(model, tok, prompt, gen_ids, device)
        sl.detach(model)

        delta_nll = st_res["nll_per_token"] - base_res["nll_per_token"]
        flips = st_res["flips_per_token"] & (~base_res["flips_per_token"])  # steered flipped from base's argmax
        
        for t in range(len(gen_ids)):
            df_rows.append({
                "method": method,
                "step": t,
                "base_nll": base_res["nll_per_token"][t],
                "steer_nll": st_res["nll_per_token"][t],
                "delta_nll": delta_nll[t],
                "cumsum_delta": delta_nll[:t+1].sum(),
                "flip": bool(flips[t])
            })
            
    df = pd.DataFrame(df_rows)
    
    # Summarize: cumulative mean dNLL up to t for log-spaced t-bins.
    # Shows shape: ramp, plateau, divergence.
    bins = [b for b in [2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096] if b <= args.max_new]
    summary = []
    for method, grp in df.groupby("method"):
        summ = {"method": method}
        for snap in bins:
            sub = grp[grp["step"] < snap]
            summ[f"dNLL@{snap}"] = sub["delta_nll"].mean()
        summary.append(summ)

    print("\nSHOULD: dNLL flat across t-bins iff steering tax is constant per token; rising iff compounding")
    print(tabulate(summary, headers="keys", tablefmt="tsv", floatfmt=".3f"))

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"trajectory__{model_id.replace('/', '--')}__seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)
    print(f"csv: {out_csv}")

    # Plot per-token dNLL (smoothed) over t in log-x.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
        for method, grp in df.groupby("method"):
            grp = grp.sort_values("step")
            t = grp["step"].values + 1
            d = grp["delta_nll"].values
            # rolling-cumulative-mean from t=1
            cm = np.cumsum(d) / np.arange(1, len(d) + 1)
            ax[0].plot(t, cm, label=method, lw=1.5)
            # per-token (smoothed via 16-window)
            w = 16
            if len(d) >= w:
                sm = np.convolve(d, np.ones(w) / w, mode="valid")
                ax[1].plot(t[w-1:], sm, label=method, lw=1.0, alpha=0.8)
        for a, title in zip(ax, ["cumulative-mean dNLL", "per-token dNLL (16-tok smoothed)"]):
            a.set_xscale("log"); a.set_xlabel("token t"); a.set_ylabel("nats/token")
            a.axhline(0.10, color="k", ls="--", lw=0.5, alpha=0.5, label="target=0.10")
            a.set_title(title); a.legend(fontsize=8); a.grid(alpha=0.3)
        fig.suptitle(f"Steering tax over trajectory | {model_id} | calib=iso-free-dNLL=0.10 | seed={args.seed}")
        fig.tight_layout()
        png = out_dir / f"trajectory__{model_id.replace('/', '--')}__seed{args.seed}.png"
        fig.savefig(png, dpi=130)
        print(f"png: {png}")
    except Exception as e:
        logger.warning(f"plot failed: {e}")

if __name__ == "__main__":
    main()
