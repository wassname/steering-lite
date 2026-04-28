"""Iso-TV calibration: find the coeff per method that produces a target mean
TV-shift on a held-out calibration set, then report leakage / SI_JS at iso-strength.

Why: comparing methods at the same nominal coeff is misleading -- additive
methods on Gemma at coeff=2.0 produce 10x smaller TV-shift than on Qwen, and
spherical lives in a totally different scale (slerp t in [0,1]). Iso-TV puts
all methods on the same "how loud is the intervention" footing.

Algorithm: extract once, then exponential search + bisection over coeff until
mean tv_target approx target_tv. ~5-7 forward passes per method.

Usage:
    python scripts/iso_tv_calibrate.py --model Qwen/Qwen3-0.6B-Base --layers 4 --target-tv 0.05
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger

import steering_lite as sl
from steering_lite.daily_dilemmas import (
    load_pairs, score_action_full, dist_metrics, make_prompt,
)


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


def base_eval(model, tok, eval_pairs, device) -> list:
    """Compute base (no steering) per-pair logp distributions and scores."""
    out = []
    for p in eval_pairs:
        rp = score_action_full(model, tok, p.situation, p.action_pos, device)
        rn = score_action_full(model, tok, p.situation, p.action_neg, device)
        out.append({"pos": rp, "neg": rn, "values_pos": p.values_pos, "values_neg": p.values_neg})
    return out


@torch.no_grad()
def demo_continuation(model, tok, situation: str, device, max_new: int = 96) -> str:
    """Greedy continuation of the dilemma prompt. Used for paper-appendix
    sanity-check of what the steered model actually generates at calibrated coeff.
    Special tokens preserved -- helps debug formatting issues."""
    prompt = make_prompt(situation)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
    out = model.generate(
        ids, max_new_tokens=max_new, do_sample=False,
        pad_token_id=tok.pad_token_id, num_return_sequences=1,
    )
    return tok.decode(out[0], skip_special_tokens=False)


@torch.no_grad()
def base_free_trajectories(model, tok, eval_pairs, device, max_new: int = 128) -> list:
    """Generate base greedy continuation for each eval pair and store base NLL
    per token. Used for free-trajectory dNLL calibration -- avoids the action-position
    overshoot problem (action calib at dNLL=0.10 -> free trajectory dNLL=0.5+)."""
    out = []
    for p in eval_pairs:
        prompt = make_prompt(p.situation)
        prompt_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=512).input_ids.to(device)
        gen = model.generate(prompt_ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        full = gen[0]
        n_p = prompt_ids.shape[1]
        # Score NLL per generated token under base
        logits = model(full.unsqueeze(0)).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        targets = full[n_p:]
        pred_logp = logp[0, n_p - 1 : full.shape[0] - 1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        out.append({
            "full_ids": full.cpu(),
            "n_prompt": n_p,
            "base_nll": (-pred_logp).cpu(),  # [n_gen]
        })
    return out


@torch.no_grad()
def steered_free_trajectory_dnll(model, tok, base_traj, device) -> dict:
    """Score steered NLL on base's free-trajectory tokens, return mean dNLL + flip rate."""
    dnll_all, flip_all = [], []
    for tr in base_traj:
        full = tr["full_ids"].to(device)
        n_p = tr["n_prompt"]
        base_nll = tr["base_nll"].to(device)
        logits = model(full.unsqueeze(0)).logits.float()
        logp = torch.log_softmax(logits, dim=-1)
        targets = full[n_p:]
        pred_logp = logp[0, n_p - 1 : full.shape[0] - 1]
        steer_nll = -pred_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        dnll = (steer_nll - base_nll)
        argmax_steer = pred_logp.argmax(dim=-1)
        flips = (argmax_steer != targets).float()
        dnll_all.append(dnll); flip_all.append(flips)
    dnll_cat = torch.cat(dnll_all)
    flip_cat = torch.cat(flip_all)
    return {
        "free_dnll": float(dnll_cat.mean()),
        "free_flip": float(flip_cat.mean()),
        "free_dnll_p95": float(dnll_cat.quantile(0.95)),
    }


def steered_eval_metrics(model, tok, eval_pairs, base_records, target_value, device,
                         base_traj=None) -> dict:
    """Run model with steering active, compare each action's logp to base_records,
    return (target_effect, leakage, tv_target, tv_other, js_target, js_other).
    If base_traj is provided, also computes free-trajectory dNLL/flip on the
    base's greedy generations (NOT action positions). This is the metric that
    actually predicts long-trajectory behavior; action-position dNLL undershoots
    free-trajectory dNLL by 1.7-5.2x with method-dependent overshoot."""
    target_l = target_value.lower().strip()
    te, le = [], []
    tv_t, tv_o = [], []
    js_t, js_o = [], []
    klba_t, klba_o = [], []
    tail_t, tail_o = [], []
    nll_t, nll_o = [], []
    flip_t, flip_o = [], []
    for p, br in zip(eval_pairs, base_records):
        rp = score_action_full(model, tok, p.situation, p.action_pos, device)
        rn = score_action_full(model, tok, p.situation, p.action_neg, device)
        for side, sr, brs, tags in (
            ("pos", rp, br["pos"], [t.lower() for t in p.values_pos]),
            ("neg", rn, br["neg"], [t.lower() for t in p.values_neg]),
        ):
            d = dist_metrics(brs["logp"], sr["logp"])
            delta = sr["score"] - brs["score"]
            if target_l in tags:
                te.append(delta); tv_t.append(d["tv"]); js_t.append(d["js"])
                klba_t.append(d["kl_ba"]); tail_t.append(d["tail_b"])
                nll_t.append(d["delta_nll"]); flip_t.append(d["flip_rate"])
            else:
                le.append(delta); tv_o.append(d["tv"]); js_o.append(d["js"])
                klba_o.append(d["kl_ba"]); tail_o.append(d["tail_b"])
                nll_o.append(d["delta_nll"]); flip_o.append(d["flip_rate"])
    def _m(xs):
        import math
        ys = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return float(sum(ys) / len(ys)) if ys else float("nan")
    out = {
        "target_effect": _m(te), "leakage": _m(le),
        "tv_target": _m(tv_t), "tv_other": _m(tv_o),
        "js_target": _m(js_t), "js_other": _m(js_o),
        "kl_ba_target": _m(klba_t), "kl_ba_other": _m(klba_o),
        "tail_target": _m(tail_t), "tail_other": _m(tail_o),
        "delta_nll_target": _m(nll_t), "delta_nll_other": _m(nll_o),
        "flip_target": _m(flip_t), "flip_other": _m(flip_o),
        "n_target": len(te), "n_other": len(le),
    }
    if base_traj is not None:
        out.update(steered_free_trajectory_dnll(model, tok, base_traj, device))
    else:
        out.update({"free_dnll": float("nan"), "free_flip": float("nan"), "free_dnll_p95": float("nan")})
    return out


def calibrate(model, tok, method: str, layers, dtype, seed: int, n_train: int,
              vectors, eval_pairs, base_records, target_value, target_metric: float,
              metric_key: str, device, tol: float = 0.01, max_iters: int = 10,
              base_traj=None) -> tuple[float, dict, list]:
    """Binary search coeff so mean(<metric_key>) approx target_metric.

    metric_key: 'tv_target' or 'delta_nll_target'. delta_nll is preferred --
    it's directly interpretable (nats of forward-NLL increase per token, which
    is the steering tax that compounds over a CoT) and bounded only above by
    coherence loss.

    For mean_diff/pca/sspace/topk_clusters/cosine_gated: coeff scales linearly
    with magnitude of the additive update, so the metric is roughly monotonic
    in |coeff|. We search on log scale.

    For spherical: coeff is slerp t in [0, 1]. Different scale, search t directly.
    """
    is_spherical = method == "spherical"
    if is_spherical:
        lo, hi = 0.001, 0.5
    else:
        lo, hi = 0.05, 16.0

    # First, find a bracket by exponential search.
    history = []
    def eval_at(c):
        s_cfg = make_cfg(method, layers, c, dtype, seed, n_train)
        sl.attach(model, s_cfg, vectors)
        m = steered_eval_metrics(model, tok, eval_pairs, base_records, target_value, device,
                                 base_traj=base_traj)
        sl.detach(model)
        history.append({"coeff": c, **m})
        logger.info(f"  [{method}] c={c:.4f} {metric_key}={m[metric_key]:.4f} "
                    f"act_dNLL={m['delta_nll_target']:.3f} free_dNLL={m['free_dnll']:.3f} "
                    f"flip_act={m['flip_target']:.2f} flip_free={m['free_flip']:.2f} "
                    f"tgt={m['target_effect']:+.3f} leak={m['leakage']:+.3f}")
        return m[metric_key]

    # Try midpoint first
    mid = (lo * hi) ** 0.5
    tv_mid = eval_at(mid)
    if tv_mid < target_metric:
        # need bigger coeff
        c = mid
        while c < hi:
            c *= 2.0
            tv = eval_at(c)
            if tv >= target_metric:
                lo, hi = c / 2.0, c
                break
        else:
            return c, history[-1], history  # hit hi bound
    else:
        c = mid
        while c > lo:
            c /= 2.0
            tv = eval_at(c)
            if tv <= target_metric:
                lo, hi = c, c * 2.0
                break
        else:
            return c, history[-1], history  # hit lo bound

    # Bisect
    for _ in range(max_iters):
        m = (lo + hi) / 2.0
        tv = eval_at(m)
        if abs(tv - target_metric) < tol:
            lo = hi = m
            break
        if tv < target_metric:
            lo = m
        else:
            hi = m

    final_c = (lo + hi) / 2.0
    final_m = history[-1]
    return final_c, final_m, history


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target", default="honesty")
    p.add_argument("--layers", default="4")
    p.add_argument("--target-tv", type=float, default=0.05, help="used if --metric=tv_target")
    p.add_argument("--target-nll", type=float, default=0.10, help="used if --metric=delta_nll_target (action) or --metric=free_dnll (free trajectory, nats/token)")
    p.add_argument("--metric", default="free_dnll", choices=["tv_target", "delta_nll_target", "free_dnll"],
                   help="calibration target. free_dnll is preferred -- predicts long-trajectory behavior. delta_nll_target (action) overshoots free trajectory by 1.7-5.2x.")
    p.add_argument("--free-traj-prompts", type=int, default=4,
                   help="#prompts for free-trajectory dNLL calibration probe. 4 prompts x 128 tok = ~512 tokens.")
    p.add_argument("--free-traj-tokens", type=int, default=128)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-eval", type=int, default=32)
    p.add_argument("--seeds", default="0", help="comma-separated seeds")
    p.add_argument("--device", default="cuda")
    p.add_argument("--torch-dtype", default="bfloat16")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--output-dir", default="outputs/iso_tv")
    args = p.parse_args()

    layers = tuple(int(x) for x in args.layers.split(","))
    dtype = getattr(torch, args.torch_dtype)
    methods = args.methods.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    target_metric = args.target_nll if args.metric in ("delta_nll_target", "free_dnll") else args.target_tv
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    logger.info(f"loading model={args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token or tok.unk_token or "<pad>"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    all_summary = []
    for seed in seeds:
        logger.info(f"########## seed={seed} ##########")
        random.seed(seed); torch.manual_seed(seed)

        pairs = load_pairs(args.target, seed=seed)
        train_pairs = pairs[: args.n_train]
        eval_pairs = pairs[args.n_train : args.n_train + args.n_eval]
        pos = [make_prompt(p.situation) + p.action_pos for p in train_pairs]
        neg = [make_prompt(p.situation) + p.action_neg for p in train_pairs]
        logger.info(f"target={args.target} n_train={len(pos)} n_eval={len(eval_pairs)}")

        logger.info("computing baseline distributions")
        base_records = base_eval(model, tok, eval_pairs, args.device)

        # Precompute base free-form trajectories (one-shot, used for all methods)
        base_traj = None
        if args.metric == "free_dnll" or args.free_traj_prompts > 0:
            n_free = min(args.free_traj_prompts, len(eval_pairs))
            logger.info(f"computing base free trajectories: {n_free} prompts x {args.free_traj_tokens} tok")
            base_traj = base_free_trajectories(model, tok, eval_pairs[:n_free], args.device,
                                               max_new=args.free_traj_tokens)

        for method in methods:
            logger.info(f"=== seed={seed} {method} ===")
            s_cfg = make_cfg(method, layers, 1.0, dtype, seed, args.n_train)
            vectors = sl.train(model, tok, pos, neg, s_cfg, batch_size=4, max_length=256)
            c, final_m, history = calibrate(
                model, tok, method, layers, dtype, seed, args.n_train,
                vectors, eval_pairs, base_records, args.target,
                target_metric, args.metric,
                args.device, base_traj=base_traj,
            )
            best = min(history, key=lambda h: abs(h[args.metric] - target_metric))
            si = best["target_effect"] - best["leakage"]
            si_tv = best["tv_target"] - best["tv_other"]
            si_js = best["js_target"] - best["js_other"]
            si_nll = best["delta_nll_target"] - best["delta_nll_other"]
            # Demo greedy continuation at calibrated coeff (seed 0 only, eval_pairs[0]).
            # Paper-appendix sanity check: are we generating coherent text?
            demo_text = ""
            demo_base = ""
            if seed == 0:
                s_cfg = make_cfg(method, layers, best["coeff"], dtype, seed, args.n_train)
                sl.attach(model, s_cfg, vectors)
                try:
                    demo_text = demo_continuation(model, tok, eval_pairs[0].situation, args.device)
                finally:
                    sl.detach(model)
                demo_base = demo_continuation(model, tok, eval_pairs[0].situation, args.device)
            row = {
                "model": args.model, "method": method, "seed": seed,
                "calibrated_coeff": best["coeff"], **best,
                "si": si, "si_tv": si_tv, "si_js": si_js, "si_nll": si_nll,
                "iters": len(history),
                "demo_steered": demo_text,
                "demo_base": demo_base,
            }
            all_summary.append(row)
            logger.info(
                f"[seed={seed} {method}] coeff*={best['coeff']:.4f} "
                f"dNLL_t={best['delta_nll_target']:.4f} dNLL_o={best['delta_nll_other']:.4f} "
                f"flip_t={best['flip_target']:.3f} tv_t={best['tv_target']:.4f} "
                f"leak={best['leakage']:+.4f} tgt={best['target_effect']:+.4f} "
                f"SI={si:+.4f} SI_NLL={si_nll:+.4f}"
            )
            if demo_text:
                logger.info(f"[demo {method}] steered:\n{demo_text[:400]}\n---")

    out_path = out_dir / f"iso__{args.model.replace('/', '--')}__L{'_'.join(map(str, layers))}__{args.metric}{target_metric}__seeds{args.seeds.replace(',', '_')}__{int(time.time())}.json"
    out_path.write_text(json.dumps({"args": vars(args), "summary": all_summary}, indent=2))
    logger.info(f"wrote {out_path}")
    print("\n| seed | method | coeff* | act_dNLL | free_dNLL | flip_free | tv_t | leakage | target | SI | SI_NLL |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in all_summary:
        print(f"| {r['seed']} | {r['method']} | {r['calibrated_coeff']:.4f} | {r['delta_nll_target']:.3f} | {r['free_dnll']:.3f} | {r['free_flip']:.3f} | {r['tv_target']:.4f} | {r['leakage']:+.4f} | {r['target_effect']:+.4f} | {r['si']:+.4f} | {r['si_nll']:+.4f} |")


if __name__ == "__main__":
    main()
