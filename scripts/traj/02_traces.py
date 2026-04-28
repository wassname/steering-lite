"""Sampled-trace demos for inspection: for each (method, alpha) pair, sample a
single completion at the alpha-scaled calibrated coeff and dump the literal text.
A stranger reading the markdown should be able to judge: coherent? gibberish?
on-topic for honesty? off the rails?

Usage:
  uv run python scripts/iso_kl/03_traces.py \\
    --iso-tv-json outputs/iso_kl/iso_kl__...json \\
    --alphas 0.5 1 4 --max-new 128 --situation-idx 0 --seed 0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 4.0])
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--situation-idx", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--chat-template", action="store_true")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--out", default=None, help="output markdown path (default outputs/trajectory/traces__...md)")
    args = ap.parse_args()

    logger.remove(); logger.add(sys.stderr, level="INFO", format="{message}")
    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = iso["args"].get("target", "honesty")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    methods = [(r["method"], r["calibrated_coeff"]) for r in rows]
    logger.info(f"model={model_id} layers={layers} alphas={args.alphas} methods={[m for m,_ in methods]}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    pairs = load_pairs(target_value, seed=args.seed)
    train_pairs = pairs[: args.n_train]
    pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
    neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
    eval_pool = pairs[args.n_train:] or pairs
    sit = eval_pool[args.situation_idx % len(eval_pool)]

    if args.chat_template:
        text = tok.apply_chat_template(
            [{"role": "user", "content": sit.situation}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=args.enable_thinking,
        )
    else:
        text = make_prompt(sit.situation)
    prompt_ids = tok(text, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)

    # Base sample
    sl.detach(model)
    torch.manual_seed(args.seed)
    base_out = model.generate(
        prompt_ids, max_new_tokens=args.max_new,
        do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
        pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
    )
    base_text = tok.decode(base_out[0, prompt_ids.shape[1]:], skip_special_tokens=False)

    md = []
    md.append(f"# Sampled trace demos\n")
    md.append(f"- model: `{model_id}`")
    md.append(f"- layers: `{layers}`")
    md.append(f"- alphas: `{args.alphas}` (multiplier on iso-calibrated coeff)")
    md.append(f"- iso source: `{Path(args.iso_tv_json).name}`")
    md.append(f"- sampling: `temp={args.temperature} top_p={args.top_p} top_k={args.top_k}`")
    md.append(f"- chat={args.chat_template} thinking={args.enable_thinking}")
    md.append(f"- max_new={args.max_new}, seed={args.seed}, situation_idx={args.situation_idx}\n")
    md.append("## Prompt\n")
    md.append("```")
    md.append(sit.situation[:600] + ("..." if len(sit.situation) > 600 else ""))
    md.append("```\n")
    md.append("## Base (no steering)\n")
    md.append("```")
    md.append(base_text)
    md.append("```\n")

    for method, coeff in methods:
        md.append(f"## {method}  (calibrated_coeff={coeff:.4f})\n")
        for alpha in args.alphas:
            scaled = coeff * alpha
            cfg = make_cfg(method, layers, scaled, dtype, args.seed, args.n_train)
            vectors = sl.train(model, tok, pos, neg, cfg, batch_size=4, max_length=256)
            sl.attach(model, cfg, vectors)
            try:
                torch.manual_seed(args.seed)
                out = model.generate(
                    prompt_ids, max_new_tokens=args.max_new,
                    do_sample=True, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                )
            finally:
                sl.detach(model)
            txt = tok.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=False)
            md.append(f"### α={alpha} (coeff={scaled:.4f})\n")
            md.append("```")
            md.append(txt)
            md.append("```\n")
            logger.info(f"{method} α={alpha} coeff={scaled:.4f} done")

    out_path = Path(args.out) if args.out else Path("outputs/trajectory") / (
        f"traces__{model_id.replace('/', '--')}__sit{args.situation_idx}__seed{args.seed}.md"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md))
    logger.info(f"wrote {out_path}")


if __name__ == "__main__":
    main()
