"""On-policy long-generation incoherence check.

For each (method, calibrated_coeff) from a prior iso-TV run, generate ONE long
completion (~256 tokens) with steering attached, compute teacher-forced PPL of
that text under the BASE model, and dump full text + special tokens to verbose.log.

The teacher-forced PPL under base = "how surprised is base by what steered said".
High PPL = steered drifted off-distribution = incoherent / hallucinatory tail.
Low PPL = steered stayed coherent (even if behaviorally shifted).

Per token-efficient-logging skill:
- BLUF result: tabulate(tsv) one row per method with PPL, cue emoji, first/last 80 chars
- Full generated text + token IDs go to logs/long_gen_verbose.log via logger.debug
- One inline SHOULD line per row

Usage:
    uv run python scripts/long_gen_check.py \
        --iso-tv-json outputs/iso_tv_n200/iso_tv__...json \
        --layers 4 --max-new 256
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


# Re-use the cfg factory from calibrate
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
def teacher_forced_ppl(base_model, tok, prompt: str, generated: str, device) -> float:
    """PPL under base of the generated continuation given prompt."""
    full = prompt + generated
    ids = tok(full, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
    n_p = p_ids.shape[1]
    if ids.shape[1] <= n_p + 1:
        return float("nan")
    logits = base_model(ids).logits.float()
    logp = torch.log_softmax(logits, dim=-1)
    targets = ids[0, n_p:ids.shape[1]]
    pred_logp = logp[0, n_p - 1:ids.shape[1] - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = -pred_logp.mean().item()
    return float(torch.tensor(nll).exp())


def cue(ppl_base: float, ppl_steer: float) -> str:
    """Cue emoji: 🟢 PPL barely shifts, 🟡 modest, 🔴 blowup."""
    if ppl_steer != ppl_steer:  # nan
        return "?"
    ratio = ppl_steer / max(ppl_base, 1e-6)
    if ratio < 1.5: return "G"   # green: coherent
    if ratio < 5.0: return "Y"   # yellow: drift
    return "R"                    # red: incoherent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iso-tv-json", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[4])
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--target-value", default=None,
                    help="defaults to args.target from iso_tv json")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0, help="which seed's coeffs to use")
    ap.add_argument("--situation-idx", type=int, default=0, help="which dilemma to use as prompt")
    ap.add_argument("--out-dir", default="outputs/long_gen")
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{message}")
    Path("logs").mkdir(exist_ok=True)
    logger.add("logs/long_gen_verbose.log", level="DEBUG", mode="w",
               format="{time:HH:mm:ss} {level} {message}")

    iso = json.load(open(args.iso_tv_json))
    model_id = iso["args"]["model"]
    target_value = args.target_value or iso["args"].get("target", "harmless")
    layers = tuple(args.layers)
    dtype = torch.bfloat16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Filter for this seed
    rows = [r for r in iso["summary"] if r["seed"] == args.seed and r["calibrated_coeff"] is not None]
    if not rows:
        logger.error(f"no rows in {args.iso_tv_json} with seed={args.seed}")
        sys.exit(1)

    logger.info(f"model={model_id} layers={layers} seed={args.seed} max_new={args.max_new}")
    logger.info(f"# methods to check: {len(rows)}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()

    # Re-extract vectors per method (need the same training pairs)
    import random
    random.seed(args.seed); torch.manual_seed(args.seed)
    pairs = load_pairs(target_value, seed=args.seed)
    train_pairs = pairs[: args.n_train]
    pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
    neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
    eval_pool = pairs[args.n_train:]
    if not eval_pool:
        eval_pool = pairs
    p = eval_pool[args.situation_idx % len(eval_pool)]
    prompt = make_prompt(p.situation)
    logger.info(f"prompt: {p.situation[:120]!r}...")

    # Base PPL of base's own greedy generation as a reference (lower bound)
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    base_gen_ids = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                   temperature=1.0, pad_token_id=tok.pad_token_id)
    base_gen = tok.decode(base_gen_ids[0, ids.shape[1]:], skip_special_tokens=False)
    base_self_ppl = teacher_forced_ppl(model, tok, prompt, base_gen, device)
    logger.debug(f"BASE_GEN: {base_gen!r}")
    logger.info(f"base self-gen PPL = {base_self_ppl:.3f} (reference floor)")

    # SHOULD: PPL of base on its own greedy gen should be ~1-3; if much higher
    # something is wrong with the chat template or tokenization.
    print(f"SHOULD: base_self_ppl in [1, 5]; got {base_self_ppl:.3f} ELSE: tokenization/template bug")

    rows_out = []
    for r in rows:
        method = r["method"]
        coeff = r["calibrated_coeff"]
        tv_cal = r["tv_target"]
        # extract for this method, seed, n_train
        cfg_train = make_cfg(method, layers, coeff, dtype, args.seed, args.n_train)
        vectors = sl.train(model, tok, pos, neg, cfg_train, batch_size=4, max_length=256)
        sl.attach(model, cfg_train, vectors)
        gen_ids = model.generate(ids, max_new_tokens=args.max_new, do_sample=False,
                                  temperature=1.0, pad_token_id=tok.pad_token_id)
        sl.detach(model)
        gen_text = tok.decode(gen_ids[0, ids.shape[1]:], skip_special_tokens=False)
        ppl = teacher_forced_ppl(model, tok, prompt, gen_text, device)
        c = cue(base_self_ppl, ppl)
        head = gen_text[:80].replace("\n", " / ").replace("\t", " ")
        tail = gen_text[-80:].replace("\n", " / ").replace("\t", " ")
        logger.debug(f"GEN[{method} c={coeff:.3f} tv={tv_cal:.3f} ppl={ppl:.2f}]: {gen_text!r}")
        logger.debug(f"GEN_IDS[{method}]: {gen_ids[0, ids.shape[1]:].tolist()}")
        rows_out.append([method, f"{coeff:.3f}", f"{tv_cal:.3f}", f"{ppl:.2f}",
                         f"x{ppl/max(base_self_ppl,1e-6):.2f}", c, head, tail])

    print()
    print(tabulate(rows_out,
                   headers=["method", "coeff", "tv_cal", "ppl", "ratio", "cue", "head", "tail"],
                   tablefmt="tsv"))
    print()
    print("# cue: G=ratio<1.5 (coherent), Y<5 (drift), R>=5 (incoherent)")
    print(f"# full text + token IDs in logs/long_gen_verbose.log")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"longgen__{model_id.replace('/', '--')}__seed{args.seed}__{args.situation_idx}.json"
    out_path.write_text(json.dumps({
        "model": model_id, "seed": args.seed, "layers": layers,
        "situation": p.situation, "base_self_ppl": base_self_ppl,
        "rows": [{"method": m, "coeff": float(c), "tv_cal": float(t), "ppl": float(pp),
                  "ratio": float(pp)/max(base_self_ppl,1e-6), "cue": cu, "head": h, "tail": tl}
                 for m, c, t, pp, _, cu, h, tl in rows_out],
    }, indent=2))
    print(f"# wrote {out_path}")


if __name__ == "__main__":
    main()
