"""Aggregate distribution shift over many prompts, on-policy steered.

For each (method, prompt) at calibrated coeff:
  Steered greedy-generates y_s of length T.
  Both base and steered score y_s -> per-position TV, KL(steer||base), KL(base||steer), NLL_diff.

We aggregate per-position metrics across prompts and save a CSV with
(method, t, n_prompts, mean_*, std_*) and a matplotlib plot.

Usage:
  uv run python scripts/aggregate_dist_shift.py \\
    --iso-tv-json outputs/iso_tv/iso__Qwen--Qwen3-0.6B-Base__L4__free_dnll0.1__seeds0_1_2__1777346635.json \\
    --n-prompts 8 --max-new 256 --seed 0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from loguru import logger
from tabulate import tabulate
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.daily_dilemmas import load_pairs, make_prompt


def make_cfg(method, layers, coeff, dtype, seed, n_train):
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
def gen(model, ids, max_new, eos_id, pad_id, sample=False, temperature=1.0, top_p=1.0, top_k=20):
    kw = dict(
        max_new_tokens=max_new, min_new_tokens=max_new,
        pad_token_id=pad_id,
        suppress_tokens=[eos_id] if eos_id is not None else None,
    )
    if sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
    else:
        kw.update(do_sample=False)
    out = model.generate(ids, **kw)
    return out[0, ids.shape[1]:]


@torch.no_grad()
def score(model, prompt_ids, gen_ids):
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
    logits = model(full).logits.float()
    n_p = len(prompt_ids)
    pred = logits[0, n_p - 1: full.shape[1] - 1]
    return torch.log_softmax(pred, dim=-1).cpu()


def per_pos_metrics(logp_b, logp_s, y_ids):
    """Return [T] tensors: tv, kl_sb, kl_bs, nll_diff, flip."""
    p_b = logp_b.exp(); p_s = logp_s.exp()
    tv = 0.5 * (p_s - p_b).abs().sum(dim=-1)
    kl_sb = (p_s * (logp_s - logp_b)).sum(dim=-1)
    kl_bs = (p_b * (logp_b - logp_s)).sum(dim=-1)
    logp_b_y = logp_b.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
    logp_s_y = logp_s.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
    nll_diff = logp_s_y - logp_b_y
    flip = (logp_b.argmax(-1) != logp_s.argmax(-1)).float()
    return tv, kl_sb, kl_bs, nll_diff, flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--n-prompts", type=int, default=8)
    ap.add_argument("--target-value", default=None)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1.0],
                    help="multipliers on calibrated coeff. e.g. 1 2 4 to see crash-off-road")
    ap.add_argument("--methods", type=str, nargs="+", default=None,
                    help="subset of methods to evaluate")
    ap.add_argument("--out-dir", default="outputs/trajectory")
    ap.add_argument("--chat-template", action="store_true",
                    help="apply tokenizer chat template (instruct models)")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="enable_thinking=True in chat template (Qwen3 thinking mode)")
    ap.add_argument("--sample", action="store_true", help="sample steered (else greedy)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()
    if args.sample:
        torch.manual_seed(args.seed)

    logger.remove(); logger.add(sys.stderr, level="INFO", format="{message}")

    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = args.target_value or iso["args"].get("target", "honesty")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    methods_coeffs = {r["method"]: r["calibrated_coeff"] for r in rows}
    if args.methods:
        methods_coeffs = {m: c for m, c in methods_coeffs.items() if m in args.methods}
    logger.info(f"BLUF: model={model_id} L={layers} N_prompts={args.n_prompts} max_new={args.max_new} alphas={args.alphas}")
    logger.info(f"methods×coeffs(α=1): {methods_coeffs}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    pairs = load_pairs(target_value, seed=args.seed)
    train_pairs = pairs[: args.n_train]
    pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
    neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
    eval_pool = pairs[args.n_train:][: args.n_prompts]
    raw_prompts = [make_prompt(p.situation) for p in eval_pool]
    if args.chat_template:
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=args.enable_thinking,
            )
            for p in raw_prompts
        ]
        logger.info(f"chat template applied (thinking={args.enable_thinking}); first prompt tail: ...{prompts[0][-200:]!r}")
    else:
        prompts = raw_prompts
    logger.info(f"got {len(prompts)} eval prompts")

    # storage: [(method, alpha)] -> list of dicts of [T] arrays
    storage = {}

    for m, base_coeff in methods_coeffs.items():
        for alpha in args.alphas:
            coeff = base_coeff * alpha
            key = (m, alpha)
            storage[key] = []
            logger.info(f"\n=== {m}  α={alpha}  coeff={coeff:.4f} ===")
            cfg = make_cfg(m, layers, coeff, dtype, args.seed, args.n_train)
            vectors = sl.train(model, tok, pos, neg, cfg, batch_size=4, max_length=256)
            for pi, prompt in enumerate(prompts):
                prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)
                sl.attach(model, cfg, vectors)
                y_s = gen(model, prompt_ids.unsqueeze(0), args.max_new, tok.eos_token_id, tok.pad_token_id,
                          sample=args.sample, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)
                steer_logp = score(model, prompt_ids, y_s)
                sl.detach(model)
                base_logp = score(model, prompt_ids, y_s)
                tv, kl_sb, kl_bs, nll_diff, flip = per_pos_metrics(base_logp, steer_logp, y_s.cpu())
                storage[key].append({
                    "tv": tv.numpy(), "kl_sb": kl_sb.numpy(), "kl_bs": kl_bs.numpy(),
                    "nll_diff": nll_diff.numpy(), "flip": flip.numpy(),
                })
                if pi == 0:
                    logger.info(f"  prompt 0 done: mean kl_sb={kl_sb.mean():.3f} flip={flip.mean():.2%}")

    # Aggregate to per-position mean & std
    rows_out = []
    for (m, alpha), plist in storage.items():
        T = min(len(p["tv"]) for p in plist)
        for t in range(T):
            for metric in ["tv", "kl_sb", "kl_bs", "nll_diff", "flip"]:
                vals = np.array([p[metric][t] for p in plist])
                rows_out.append({
                    "method": m, "alpha": alpha, "t": t, "metric": metric,
                    "mean": float(vals.mean()), "std": float(vals.std()),
                    "p10": float(np.quantile(vals, 0.10)),
                    "p50": float(np.quantile(vals, 0.50)),
                    "p90": float(np.quantile(vals, 0.90)),
                    "n": len(vals),
                })
    df = pd.DataFrame(rows_out)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = "_".join(f"a{a}" for a in args.alphas)
    csv = out_dir / f"agg_dist_shift__{model_id.replace('/', '--')}__N{args.n_prompts}__T{args.max_new}__{alpha_tag}__seed{args.seed}.csv"
    df.to_csv(csv, index=False)
    logger.info(f"\ncsv: {csv}")

    # Plot: 1 row (KL_sb only) × n_alphas cols, p50 line + p10..p90 band
    n_alphas = len(args.alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(5.5 * n_alphas, 4.5), squeeze=False)
    colors = plt.cm.tab10.colors
    method_color = {m: colors[i] for i, m in enumerate(methods_coeffs)}
    window = 4
    metric, ylabel, log = "kl_sb", "KL(steer || base) [nats]", True
    for j, alpha in enumerate(args.alphas):
        ax = axes[0, j]
        for m in methods_coeffs:
            sub = df[(df.method == m) & (df.alpha == alpha) & (df.metric == metric)].sort_values("t")
            if len(sub) == 0: continue
            t = sub["t"].values
            p50 = np.convolve(sub["p50"].values, np.ones(window)/window, mode="same")
            p10 = np.convolve(sub["p10"].values, np.ones(window)/window, mode="same")
            p90 = np.convolve(sub["p90"].values, np.ones(window)/window, mode="same")
            lab = f"{m} (c={methods_coeffs[m]*alpha:.2g})" if j == 0 else None
            ax.plot(t, p50, label=lab, color=method_color[m], lw=1.4)
            ax.fill_between(t, np.maximum(p10, 1e-3) if log else p10, p90, alpha=0.12, color=method_color[m])
        ax.set_xlabel("token position t")
        ax.set_ylabel(ylabel)
        ax.set_title(f"α = {alpha}× calibrated coeff")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 1e1)
        ax.axhline(1.0, color="black", lw=1.6, alpha=0.85, zorder=5)
        ax.axvline(20, color="black", lw=1.0, ls=":", alpha=0.7, zorder=5)
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    iso_args = iso.get("args", {})
    iso_label = (
        f"{iso_args.get('mode', 'free-dNLL')}_{iso_args.get('target_stat', '')}={iso_args.get('target_kl', iso_args.get('target_metric_value', '?'))}"
        if 'mode' in iso_args else f"free-dNLL={iso_args.get('target_metric_value', '?')}"
    )
    fig.suptitle(f"Distribution drift along the trajectory · KL(steer∥base) by token position · {model_id}  layer={layers[0]}  N={args.n_prompts} prompts · iso-calibration: {iso_label}\nsolid p50, band p10–p90, dotted t={iso_args.get('t_calib', 20)} (calibration window), heavy line KL=1", fontsize=9)
    fig.tight_layout()
    png = out_dir / f"agg_dist_shift__{model_id.replace('/', '--')}__N{args.n_prompts}__T{args.max_new}__{alpha_tag}__seed{args.seed}.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    logger.info(f"png: {png}")

    # Concise table: mean kl_sb at last 16 tokens minus first 16 tokens (slope sign), per alpha
    print("\n=== mean kl_sb at early (t∈[16,32)) vs late (t∈[T-32,T-16)) per α — positive Δ = drift up ===")
    early_lo, early_hi = 16, min(48, args.max_new)
    late_hi = args.max_new
    late_lo = max(0, late_hi - 32)
    rows_tab = []
    for m in methods_coeffs:
        for alpha in args.alphas:
            sub = df[(df.method == m) & (df.alpha == alpha) & (df.metric == "kl_sb")]
            if len(sub) == 0: continue
            early = sub[(sub.t >= early_lo) & (sub.t < early_hi)]["mean"].mean()
            late = sub[(sub.t >= late_lo) & (sub.t < late_hi)]["mean"].mean()
            sub_f = df[(df.method == m) & (df.alpha == alpha) & (df.metric == "flip")]
            mean_flip = sub_f["mean"].mean()
            rows_tab.append({
                "method": m, "α": alpha, "coeff": f"{methods_coeffs[m]*alpha:.3f}",
                "kl_sb early": f"{early:.3f}", "kl_sb late": f"{late:.3f}",
                "Δ kl_sb": f"{late - early:+.3f}", "flip rate": f"{mean_flip:.1%}",
            })
    print(tabulate(rows_tab, headers="keys", tablefmt="tsv"))
    print(f"\nSHOULD: at α=1, kl_sb early≈late (calibrated to be safe). At α=4, Δ kl_sb >> 0 OR mean kl_sb explodes (>1.0) ELSE method has no crash regime in this T window.")


if __name__ == "__main__":
    main()
