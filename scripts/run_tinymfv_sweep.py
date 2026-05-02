"""tinymfv sweep: extract -> calibrate -> eval, one row per method.

Persona-branching contrastive pairs (POS/NEG share suffix, differ in persona).
Every method is iso-KL calibrated. Eval: tinymfv guided CoT, 64 think tokens.

Refs:
  calibration: https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7
  guided gen: https://gist.github.com/wassname/733c568cd29c2a402be4442d6a061899
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite._quiet import quiet_external_logs
from steering_lite.data import make_persona_pairs
from steering_lite.eval.tinymfv import evaluate_with_vector

quiet_external_logs()
logger.remove()
logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=False, format="{message}")


METHODS = [
    "mean_diff", "mean_centred", "pca", "topk_clusters", "cosine_gated",
    "sspace", "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering",
]


def _make_cfg(method: str, layers: tuple[int, ...]) -> sl.SteeringConfig:
    common = dict(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0)
    table = {
        "mean_diff":             sl.MeanDiffC(**common),
        "mean_centred":          sl.MeanDiffC(**common, subtract_corpus_mean=True),
        "pca":                   sl.PCAC(**common),
        "topk_clusters":         sl.TopKClustersC(**common, k=4),
        "cosine_gated":          sl.CosineGatedC(**common, tau=0.0),
        "sspace":                sl.SSpaceC(**common, r=8),
        "spherical":             sl.SphericalC(**common),
        "directional_ablation":  sl.DirectionalAblationC(**common),
        "chars":                 sl.CHaRSC(**common, k=4),
        "linear_act":            sl.LinearAcTC(**common),
        "angular_steering":      sl.AngularSteeringC(**common),
    }
    return table[method]


def _resolve_layers(model, layers_arg: str) -> tuple[int, ...]:
    n = model.config.num_hidden_layers
    if layers_arg == "mid":
        # central 50%
        lo, hi = n // 4, (3 * n) // 4
        return tuple(range(lo, hi))
    return tuple(int(x) for x in layers_arg.split(","))


def _calib_prompts(tok, n: int = 8, seed: int = 0) -> list[str]:
    """Held-out user_msgs (no persona) for KL measurement."""
    from steering_lite.data import load_suffixes
    import random
    rng = random.Random(seed)
    entries = load_suffixes(thinking=True)
    rng.shuffle(entries)
    seen = set()
    out = []
    for e in entries:
        if e["user_msg"] in seen:
            continue
        seen.add(e["user_msg"])
        out.append(e["user_msg"])
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--n-pairs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--target-kl", type=float, default=1.0)
    ap.add_argument("--calib-T", type=int, default=20)
    ap.add_argument("--calib-iters", type=int, default=8)
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--vignettes", default="scifi")
    ap.add_argument("--out", type=Path, default=Path("outputs/tinymfv_sweep"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    logger.info(f"BLUF: model={args.model} methods={len(args.methods)} target_kl={args.target_kl} "
                f"vignettes={args.vignettes} max_think={args.max_think_tokens}")
    logger.info("EXPECT: extract -> calibrate (iso-KL) -> eval. "
                "One demo trace per stage shows decoded prompt+gen with special tokens. "
                "Δ_target = mean(p_true_steer) - mean(p_true_base); sign tracked post-hoc.")

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    pos_prompts, neg_prompts = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True)
    calib_prompts = _calib_prompts(tok, n=8)

    logger.info("\n=== Baseline eval (no steering) ===")
    base_t0 = time.time()
    base_report = evaluate_with_vector(model, tok, name=args.vignettes,
                                       max_think_tokens=args.max_think_tokens)
    base_p_true_mean = float(sum(base_report.get("p_true", [])) / max(1, len(base_report.get("p_true", [])))) \
        if "p_true" in base_report else float(base_report["info"].get("p_true_mean", 0.0))
    logger.info(f"baseline elapsed={time.time()-base_t0:.1f}s")

    rows: list[list] = []
    for method in tqdm(args.methods, desc="methods", mininterval=60):
        try:
            cfg = _make_cfg(method, layers)
        except KeyError:
            logger.warning(f"skip unknown method {method!r}")
            continue
        logger.info(f"\n=== {method} ===")
        t0 = time.time()
        v = sl.train(model, tok, pos_prompts, neg_prompts, cfg,
                     batch_size=args.batch_size, max_length=args.max_length)
        coeff_calib, _hist = sl.calibrate_iso_kl(
            v, model, tok, calib_prompts,
            target_kl=args.target_kl, T=args.calib_T,
            max_iters=args.calib_iters, device=args.device,
        )
        v.cfg.coeff = float(coeff_calib)
        kl_hit = _hist[-1].get("kl_p95", float("nan")) if _hist else float("nan")

        with v(model):
            steer_report = evaluate_with_vector(
                model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens)
        steer_p_true_mean = float(sum(steer_report.get("p_true", [])) / max(1, len(steer_report.get("p_true", [])))) \
            if "p_true" in steer_report else float(steer_report["info"].get("p_true_mean", 0.0))

        delta = steer_p_true_mean - base_p_true_mean
        cue = "🟢" if abs(delta) > 0.05 else ("🟡" if abs(delta) > 0.01 else "🔴")
        elapsed = time.time() - t0
        rows.append([cue, f"{delta:+.3f}", method, f"{coeff_calib:+.3f}",
                     f"{kl_hit:.2f}", f"{base_p_true_mean:.3f}", f"{steer_p_true_mean:.3f}",
                     f"{elapsed:.0f}s"])

        out_path = args.out / f"{method}.json"
        out_path.write_text(json.dumps({
            "method": method, "model": args.model, "layers": list(layers),
            "coeff_calibrated": float(coeff_calib), "target_kl": args.target_kl,
            "kl_p95_at_calib": kl_hit,
            "base_p_true_mean": base_p_true_mean,
            "steer_p_true_mean": steer_p_true_mean, "delta": delta,
            "n_pairs": args.n_pairs, "max_think_tokens": args.max_think_tokens,
            "vignettes": args.vignettes, "elapsed_s": elapsed,
        }, indent=2))

    logger.info("\n=== tinymfv sweep complete ===")
    logger.info(f"out: {args.out}")
    logger.info("\n" + tabulate(
        rows,
        headers=["cue", "Δ_target", "method", "C_calib", "kl_p95", "p_b", "p_s", "t"],
        tablefmt="tsv",
    ))


if __name__ == "__main__":
    main()
