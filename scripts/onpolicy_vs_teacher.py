"""On-policy vs teacher-forced trajectory comparison.

For each method at its calibrated coeff:
  TF-base:  base greedy-generates y_b. Steered scores y_b. "Did steering disrupt base's path?"
  OP-steer: steered greedy-generates y_s. Base scores y_s. "Did steered drift to gibberish?"

Per position t we compute (over the full vocab, no sampling noise):
  TV_t   = 0.5 * sum_v |p_a(v) - p_b(v)|
  KL_t   = sum_v p_a(v) log(p_a(v) / p_b(v))      (a is "from", b is "to"; KL(steer||base) under TF-base regime)
  NLL_d  = -log p_b(y_t) - (-log p_a(y_t))         (single-sample estimate)

Reports mean over t in length-bins {2,4,16,32,64,128}.

Usage:
  uv run python scripts/onpolicy_vs_teacher.py \\
    --iso-tv-json outputs/iso_tv/iso__Qwen--Qwen3-0.6B-Base__L4__free_dnll0.1__seeds0__1777343934.json \\
    --layers 4 --max-new 128 --seed 0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

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
def greedy_gen(model, ids, max_new, eos_id, pad_id):
    """Greedy generate max_new tokens, suppress EOS so we always get full length."""
    out = model.generate(
        ids, max_new_tokens=max_new, min_new_tokens=max_new,
        do_sample=False, pad_token_id=pad_id,
        suppress_tokens=[eos_id] if eos_id is not None else None,
    )
    return out[0, ids.shape[1]:]


@torch.no_grad()
def score_trajectory(model, prompt_ids, gen_ids):
    """Forward model on prompt+gen, return per-token logprobs + full distributions
    aligned to the gen tokens.
    Returns:
      logp_at_y: [T] log p_model(y_t | ctx_<t)
      logp_full: [T, V] log_softmax over vocab predicting positions 1..T
    """
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
    logits = model(full).logits.float()  # [1, P+T, V]
    n_p = len(prompt_ids)
    # Position p_p-1 predicts y_0, p_p predicts y_1, ...
    pred_logits = logits[0, n_p - 1 : full.shape[1] - 1]  # [T, V]
    logp_full = torch.log_softmax(pred_logits, dim=-1)
    logp_at_y = logp_full.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
    return logp_at_y.cpu(), logp_full.cpu()


def metrics_per_position(logp_a, logp_b, y_ids):
    """logp_a, logp_b: [T, V] log-softmaxed. y_ids: [T].
    Returns dict of [T] tensors:
      tv:   0.5 * sum_v |p_a - p_b|
      kl_ab: sum_v p_a (log p_a - log p_b) = KL(a || b)
      nll_diff: (-log p_b(y)) - (-log p_a(y))   = log p_a(y) - log p_b(y)
    """
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    tv = 0.5 * (p_a - p_b).abs().sum(dim=-1)
    # KL(a || b)
    kl_ab = (p_a * (logp_a - logp_b)).sum(dim=-1)
    # NLL difference at sampled token
    logp_a_y = logp_a.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
    logp_b_y = logp_b.gather(-1, y_ids.unsqueeze(-1)).squeeze(-1)
    nll_diff = logp_a_y - logp_b_y  # equivalently NLL_b - NLL_a
    return {"tv": tv, "kl_ab": kl_ab, "nll_diff": nll_diff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--target-value", default=None)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--situation-idx", type=int, default=0)
    ap.add_argument("--out-dir", default="outputs/trajectory")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")

    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = args.target_value or iso["args"].get("target", "honesty")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    if not rows:
        logger.error(f"no rows in {args.iso_tv_json} with seed={args.seed}"); sys.exit(1)

    logger.info(f"model={model_id} layers={layers} seed={args.seed} max_new={args.max_new}")
    logger.info(f"methods: {[r['method'] for r in rows]}")

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
    eval_pool = pairs[args.n_train:] or pairs
    sit = eval_pool[args.situation_idx % len(eval_pool)]
    prompt = make_prompt(sit.situation)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)

    # --- Base trajectory (TF-base regime) ---
    logger.info("base greedy gen...")
    sl.detach(model)
    y_b = greedy_gen(model, prompt_ids.unsqueeze(0), args.max_new, tok.eos_token_id, tok.pad_token_id)
    base_text = tok.decode(y_b, skip_special_tokens=False)
    logger.debug(f"BASE TRAJ: {base_text!r}")

    base_logp_at_yb, base_logp_full_yb = score_trajectory(model, prompt_ids, y_b)

    bins = [2, 4, 16, 32, 64, 128]
    bins = [b for b in bins if b <= args.max_new]
    rows_out = []
    text_demos = {}

    for r in rows:
        method = r["method"]
        coeff = r["calibrated_coeff"]
        cfg = make_cfg(method, layers, coeff, dtype, args.seed, args.n_train)
        vectors = sl.train(model, tok, pos, neg, cfg, batch_size=4, max_length=256)

        # --- TF-base: score y_b under steered ---
        sl.attach(model, cfg, vectors)
        st_logp_at_yb, st_logp_full_yb = score_trajectory(model, prompt_ids, y_b)
        # --- OP-steer: steered generates y_s ---
        y_s = greedy_gen(model, prompt_ids.unsqueeze(0), args.max_new, tok.eos_token_id, tok.pad_token_id)
        st_text = tok.decode(y_s, skip_special_tokens=False)
        st_logp_at_ys, st_logp_full_ys = score_trajectory(model, prompt_ids, y_s)
        sl.detach(model)
        # base scores y_s (off-policy for base)
        base_logp_at_ys, base_logp_full_ys = score_trajectory(model, prompt_ids, y_s)

        text_demos[method] = {"base": base_text, "steer": st_text}

        # TF-base: a=steer, b=base on y_b
        tf = metrics_per_position(st_logp_full_yb, base_logp_full_yb, y_b.cpu())
        # OP-steer: a=steer, b=base on y_s (now y_s is what steered would write; base is "off-policy" judge)
        op = metrics_per_position(st_logp_full_ys, base_logp_full_ys, y_s.cpu())

        for snap in bins:
            for regime, m in [("TF_base", tf), ("OP_steer", op)]:
                rows_out.append({
                    "method": method,
                    "regime": regime,
                    "len": snap,
                    "tv": float(m["tv"][:snap].mean()),
                    "kl": float(m["kl_ab"][:snap].mean()),
                    "nll_diff": float(m["nll_diff"][:snap].mean()),
                })

    df = pd.DataFrame(rows_out)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"onpolicy_vs_tf__{model_id.replace('/', '--')}__seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)

    # Pivot for readability: one table per (regime, metric)
    print("\n=== TF-base (base writes y_b, steered scores) ===")
    for metric in ["tv", "kl", "nll_diff"]:
        sub = df[df.regime == "TF_base"].pivot(index="method", columns="len", values=metric)
        print(f"\n--- {metric} ---")
        print(tabulate(sub, headers="keys", tablefmt="tsv", floatfmt=".4f"))

    print("\n=== OP-steer (steered writes y_s, base scores) ===")
    for metric in ["tv", "kl", "nll_diff"]:
        sub = df[df.regime == "OP_steer"].pivot(index="method", columns="len", values=metric)
        print(f"\n--- {metric} ---")
        print(tabulate(sub, headers="keys", tablefmt="tsv", floatfmt=".4f"))

    # Demo trajectories
    print("\n=== Demo trajectories (first 240 chars) ===")
    for m, td in text_demos.items():
        print(f"\n[{m}] BASE: {td['base'][:240]!r}")
        print(f"[{m}] STEER: {td['steer'][:240]!r}")

    print(f"\ncsv: {out_csv}")
    out_demo = out_dir / f"onpolicy_vs_tf__{model_id.replace('/', '--')}__seed{args.seed}__demos.json"
    json.dump(text_demos, open(out_demo, "w"), indent=2)
    print(f"demos: {out_demo}")

    print("\nSHOULD: TF_base.kl ≈ calibration target (0.10 nats); OP_steer.kl can be much larger if steered drifts to gibberish.")
    print("SHOULD: nll_diff ≈ kl in expectation on TF_base (single-sample estimate); on OP_steer they may diverge since steered's argmax is high under steered but maybe low under base.")


if __name__ == "__main__":
    main()
