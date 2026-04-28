"""Distribution shift at exponential token positions, for ONE example.

For each method at its calibrated coeff, we dump:
  - Literal base text y_b
  - Literal steered text y_s
  - At positions t in {1,2,4,8,16,32,64,128}, both regimes:
      TF_base regime  (base wrote y_b; both models score y_b at position t):
        token = y_b[t], p_base(token), p_steer(token), TV_t, KL(steer||base)_t, KL(base||steer)_t,
        argmax_base, argmax_steer, flipped?
      OP_steer regime (steered wrote y_s; both models score y_s at position t):
        same fields with y_s

Goal: a stranger can read a single artifact and SEE the distribution drift along
the trajectory + the literal text it was sampling. No aggregate hiding tail events.

Usage:
  uv run python scripts/expo_traj_demo.py \\
    --iso-tv-json outputs/iso_tv/iso__Qwen--Qwen3-0.6B-Base__L4__free_dnll0.1__seeds0__1777343934.json \\
    --max-new 128 --situation-idx 0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
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
def greedy_gen(model, ids, max_new, eos_id, pad_id):
    out = model.generate(
        ids, max_new_tokens=max_new, min_new_tokens=max_new,
        do_sample=False, pad_token_id=pad_id,
        suppress_tokens=[eos_id] if eos_id is not None else None,
    )
    return out[0, ids.shape[1]:]


@torch.no_grad()
def score(model, prompt_ids, gen_ids):
    """Returns logp_full [T, V] aligned to gen_ids predictions."""
    full = torch.cat([prompt_ids, gen_ids]).unsqueeze(0)
    logits = model(full).logits.float()
    n_p = len(prompt_ids)
    pred = logits[0, n_p - 1: full.shape[1] - 1]
    return torch.log_softmax(pred, dim=-1).cpu()


def position_table(logp_b, logp_s, y_ids, tok, positions):
    """For each position t in `positions`, build a row with token + dist-shift fields.
    logp_b, logp_s: [T, V] log-softmax. y_ids: [T] long.
    """
    p_b = logp_b.exp(); p_s = logp_s.exp()
    rows = []
    for t in positions:
        if t >= len(y_ids):
            continue
        # KL needs careful eps for KL(b||s) where p_b can be ~0
        kl_sb = float((p_s[t] * (logp_s[t] - logp_b[t])).sum())  # KL(steer||base)
        kl_bs = float((p_b[t] * (logp_b[t] - logp_s[t])).sum())  # KL(base||steer)
        tv = float(0.5 * (p_s[t] - p_b[t]).abs().sum())
        amax_b = int(logp_b[t].argmax())
        amax_s = int(logp_s[t].argmax())
        y_t = int(y_ids[t])
        rows.append({
            "t": t,
            "tok": tok.decode([y_t]).replace("\n", "\\n"),
            "p_b": float(p_b[t, y_t]),
            "p_s": float(p_s[t, y_t]),
            "tv": tv,
            "kl_sb": kl_sb,
            "kl_bs": kl_bs,
            "amax_b": tok.decode([amax_b]).replace("\n", "\\n"),
            "amax_s": tok.decode([amax_s]).replace("\n", "\\n"),
            "flipped": amax_b != amax_s,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--target-value", default=None)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--situation-idx", type=int, default=0)
    ap.add_argument("--method", default=None, help="restrict to one method (else all)")
    ap.add_argument("--out-dir", default="outputs/trajectory")
    args = ap.parse_args()

    logger.remove(); logger.add(sys.stderr, level="INFO", format="{message}")

    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = args.target_value or iso["args"].get("target", "honesty")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    if args.method:
        rows = [r for r in rows if r["method"] == args.method]
    if not rows:
        logger.error("no rows"); sys.exit(1)

    logger.info(f"BLUF: model={model_id} L={layers} seed={args.seed} max_new={args.max_new} situation={args.situation_idx}")
    logger.info(f"methods: {[r['method'] for r in rows]}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    pairs = load_pairs(target_value, seed=args.seed)
    train_pairs = pairs[: args.n_train]
    pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
    neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
    eval_pool = pairs[args.n_train:] or pairs
    sit = eval_pool[args.situation_idx % len(eval_pool)]
    prompt = make_prompt(sit.situation)
    prompt_ids = tok(prompt, return_tensors="pt").input_ids[0].to(device)

    logger.info(f"PROMPT (last 200 chars): ...{prompt[-200:]!r}")

    sl.detach(model)
    y_b = greedy_gen(model, prompt_ids.unsqueeze(0), args.max_new, tok.eos_token_id, tok.pad_token_id)
    base_logp_yb = score(model, prompt_ids, y_b)
    base_text = tok.decode(y_b, skip_special_tokens=False)

    positions = [p for p in [1, 2, 4, 8, 16, 32, 64, 128] if p < args.max_new]

    artifact = {
        "model": model_id,
        "prompt_tail": prompt[-400:],
        "max_new": args.max_new,
        "positions": positions,
        "base_text": base_text,
        "methods": {},
    }

    for r in rows:
        method = r["method"]; coeff = r["calibrated_coeff"]
        logger.info(f"\n{'='*72}\n=== {method}  coeff={coeff:.4f}  iso_target={iso['args'].get('target_metric_value', '?')} ===\n{'='*72}")
        cfg = make_cfg(method, layers, coeff, dtype, args.seed, args.n_train)
        vectors = sl.train(model, tok, pos, neg, cfg, batch_size=4, max_length=256)

        # TF_base: score y_b under steered
        sl.attach(model, cfg, vectors)
        steer_logp_yb = score(model, prompt_ids, y_b)
        # OP_steer: steered generates
        y_s = greedy_gen(model, prompt_ids.unsqueeze(0), args.max_new, tok.eos_token_id, tok.pad_token_id)
        steer_text = tok.decode(y_s, skip_special_tokens=False)
        steer_logp_ys = score(model, prompt_ids, y_s)
        sl.detach(model)
        base_logp_ys = score(model, prompt_ids, y_s)

        tf_rows = position_table(base_logp_yb, steer_logp_yb, y_b.cpu(), tok, positions)
        op_rows = position_table(base_logp_ys, steer_logp_ys, y_s.cpu(), tok, positions)

        print(f"\n--- {method}: BASE TEXT (y_b, what base greedy emits) ---")
        print(repr(base_text))
        print(f"\n--- {method}: STEER TEXT (y_s, what steered greedy emits, coeff={coeff:.4f}) ---")
        print(repr(steer_text))

        print(f"\n--- {method}: TF_base regime  (token=y_b[t]; both models score the SAME base trajectory) ---")
        print("SHOULD: tv,kl small for safe coeffs. flipped=True means steered would have decoded differently here.")
        print(tabulate(tf_rows, headers="keys", tablefmt="tsv", floatfmt=".4f"))

        print(f"\n--- {method}: OP_steer regime (token=y_s[t]; both models score the steered trajectory) ---")
        print("SHOULD: kl_bs (base||steer) large means base assigns ~0 prob to steer's tokens (gibberish detector).")
        print(tabulate(op_rows, headers="keys", tablefmt="tsv", floatfmt=".4f"))

        # summary numbers
        n_flipped_tf = sum(r["flipped"] for r in tf_rows)
        n_flipped_op = sum(r["flipped"] for r in op_rows)
        print(f"\nTLDR {method}: TF flipped={n_flipped_tf}/{len(tf_rows)} positions, OP flipped={n_flipped_op}/{len(op_rows)} positions")
        print(f"     mean kl_sb (TF)={sum(r['kl_sb'] for r in tf_rows)/len(tf_rows):.3f}  mean kl_bs (OP)={sum(r['kl_bs'] for r in op_rows)/len(op_rows):.3f}")

        artifact["methods"][method] = {
            "coeff": coeff,
            "steer_text": steer_text,
            "tf_base": tf_rows,
            "op_steer": op_rows,
        }

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"expo__{model_id.replace('/', '--')}__seed{args.seed}__sit{args.situation_idx}.json"
    json.dump(artifact, open(out, "w"), indent=2, default=str)
    print(f"\nartifact: {out}")
    print("SHOULD final: at least one method has any flipped=True at t>=8 ELSE coeff is too small or method has no concept direction.")


if __name__ == "__main__":
    main()
