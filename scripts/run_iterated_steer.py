"""Iterated steering: accumulate a sum of fresh vectors, plot trajectory.

mean_diff only -- the math is purely linear, so the "stacked" vector is just
`sum_i (sign_i * C_i * v_i_unit)`, accumulated externally via `Vector + Vector`.
No in-hook stacking, no per-slot machinery. sspace iteration is parked (the
cosine gate is nonlinear, so summing N gated deltas != one gated delta).

Each round:
  1. Extract v_fresh under the currently-attached v_running. The new contrast
     captures whatever signal remains after prior rounds.
  2. Iso-KL calibrate |C| in isolation against the bare model (attach.py is
     single-slot, so we can't have v_running attached during calibration).
     Per-round target_kl=1.0; total drift accumulates.
  3. Bake C into v_fresh's state (`v_fresh = v_fresh * C`, set coeff=1).
  4. Eval at v_running + v_fresh and v_running - v_fresh, pick sign with
     lower Authority Δlogit. Commit: `v_running += sign * v_fresh`.

Outputs:
  - rounds.tsv         BLUF table: round, ±, axis, dlogit_<F>, pmass, ppl
  - round_NN.json      raw_p_true / raw_pmass / raw_nll for offline plotting
  - meta.json          full run config + rounds summary
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from loguru import logger
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite._quiet import quiet_external_logs
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY
from steering_lite.eval.tinymfv import evaluate_with_vector
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, FOUNDATION_SHORT,
    baseline_logit_per_foundation, dlogit_per_foundation,
    axis_shift, _mean_pmass,
)
from steering_lite.vector import Vector
from _meta import make_metadata, append_run

quiet_external_logs()
logger.remove()
logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=False, format="{message}")


def _resolve_layers(model, layers_arg: str) -> tuple[int, ...]:
    n = model.config.num_hidden_layers
    if layers_arg == "mid":
        lo = max(2, int(n * 0.2))
        hi = min(n - 2, int(n * 0.8))
        return tuple(range(lo, hi))
    return tuple(int(x) for x in layers_arg.split(","))


def _calib_prompts(tok, n: int = 8, seed: int = 0) -> list[str]:
    from steering_lite.data import load_suffixes
    import random
    rng = random.Random(seed)
    entries = load_suffixes(thinking=True)
    rng.shuffle(entries)
    seen, out = set(), []
    for e in entries:
        if e["user_msg"] in seen:
            continue
        seen.add(e["user_msg"])
        out.append(e["user_msg"])
        if len(out) >= n:
            break
    return out


def _ppl(report) -> float:
    nll = report["info"].get("prompt_nll_mean")
    return math.exp(nll) if nll is not None and not math.isnan(nll) else float("nan")


def _bake_coeff(v: Vector) -> Vector:
    """Fold v.cfg.coeff into state, return new Vector with coeff=1.

    mean_diff state is `{v: tensor}`; scaling all buffers is exact for linear
    methods. After this, `v_running + v_fresh` directly sums vectors regardless
    of when each was calibrated.
    """
    scaled = v * float(v.cfg.coeff)
    scaled.cfg.coeff = 1.0
    return scaled


def _auth_logit(dlogit_per_f) -> float:
    return dlogit_per_f["Authority"]["mean"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--method", choices=["mean_diff"], default="mean_diff",
                    help="iterated steering is mean_diff only (linear math). "
                         "sspace's gate is nonlinear, so sum-of-vectors != "
                         "stacked-gated-deltas; parked.")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--n-pairs", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--target-kl", type=float, default=1.0)
    ap.add_argument("--calib-T", type=int, default=20)
    ap.add_argument("--calib-iters", type=int, default=8)
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--vignettes", default="airisk")
    ap.add_argument("--out", type=Path, default=Path("outputs/iterated_steer"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(args)
    logger.info(
        f"BLUF: iterated {args.method}. rounds={args.rounds} model={args.model} "
        f"target_kl={args.target_kl} vignettes={args.vignettes}"
    )
    logger.info(
        "SHOULD: Authority Δlogit drops further each round (cumulative); pmass and "
        "perplexity degrade gradually. ELSE: signal exhausted (round 2 saw nothing) "
        "or model went OOD (pmass<0.5)."
    )

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    # Round 0: bare reference.
    logger.info("\n=== round 0 (bare) ===")
    t0 = time.time()
    base_report = evaluate_with_vector(model, tok, name=args.vignettes,
                                       max_think_tokens=args.max_think_tokens)
    base_logit_per_f = baseline_logit_per_foundation(base_report, args.vignettes)
    base_pmass = _mean_pmass(base_report)
    base_ppl = _ppl(base_report)
    logger.info(
        f"bare pmass={base_pmass:.3f} ppl={base_ppl:.2f} "
        f"Auth={base_logit_per_f['Authority']['mean']:+.3f}±{base_logit_per_f['Authority']['std']:.2f} "
        f"({time.time() - t0:.1f}s)"
    )

    pos_prompts, neg_prompts = make_persona_pairs(
        tok, n_pairs=args.n_pairs, thinking=True,
        persona_pairs=PERSONA_PAIRS_AUTHORITY,
    )
    calib_prompts = _calib_prompts(tok, n=8)

    rows: list[list] = []
    rows.append([
        "0", "—", "+0.00",
        *(f"{base_logit_per_f[f]['mean']:+.2f}" for f in FOUNDATION_ORDER),
        f"{base_pmass:.3f}", f"{base_ppl:.2f}", "0",
    ])

    v_running: Vector | None = None
    round_summaries = []

    def _make_cfg() -> sl.SteeringConfig:
        return sl.MeanDiffC(layers=layers, coeff=1.0, dtype=dtype, seed=0)

    for r in range(1, args.rounds + 1):
        logger.info(f"\n=== round {r} ===")
        t_round = time.time()

        # Extract under v_running. If attached, its hook modifies the residual
        # that record_activations sees, so the new contrast captures residual
        # signal after prior rounds.
        if v_running is None:
            v_fresh = sl.train(model, tok, pos_prompts, neg_prompts, _make_cfg(),
                               batch_size=args.batch_size, max_length=args.max_length)
        else:
            with v_running(model):
                v_fresh = sl.train(model, tok, pos_prompts, neg_prompts, _make_cfg(),
                                   batch_size=args.batch_size, max_length=args.max_length)

        # Calibrate v_fresh in isolation vs bare model. attach.py is single-slot,
        # so we can't have v_running attached during calibration. Per-round KL is
        # "round-r alone vs bare", not "round-r vs (bare + v_running)".
        coeff_calib, hist = sl.calibrate_iso_kl(
            v_fresh, model, tok, calib_prompts,
            target_kl=args.target_kl, T=args.calib_T,
            max_iters=args.calib_iters, device=args.device,
        )
        kl_hit = hist[-1].get("kl_p95", float("nan")) if hist else float("nan")
        C = float(coeff_calib)
        logger.info(f"  calibrated |C|={C:.4f} kl_p95={kl_hit:.2f}")

        # Bake C into state so v_fresh.cfg.coeff=1 -- now Vector + Vector sums
        # magnitudes directly without coeff bookkeeping.
        v_fresh = _bake_coeff(v_fresh)

        # Eval ± at v_running + (±1) * v_fresh.
        pos_dlogit, pos_report = _eval_with(v_running, +1.0 * v_fresh, model, tok, args, base_report)
        neg_dlogit, neg_report = _eval_with(v_running, -1.0 * v_fresh, model, tok, args, base_report)
        auth_pos, auth_neg = _auth_logit(pos_dlogit), _auth_logit(neg_dlogit)
        if auth_pos <= auth_neg:
            sign, signed_C, chosen_dlogit, chosen_report = "+", +C, pos_dlogit, pos_report
            signed_fresh = +1.0 * v_fresh
        else:
            sign, signed_C, chosen_dlogit, chosen_report = "-", -C, neg_dlogit, neg_report
            signed_fresh = -1.0 * v_fresh
        logger.info(f"  Auth Δ: +C={auth_pos:+.3f}  -C={auth_neg:+.3f}  → chose {sign}C")

        v_running = signed_fresh if v_running is None else v_running + signed_fresh

        round_pmass = _mean_pmass(chosen_report)
        round_ppl = _ppl(chosen_report)
        elapsed = time.time() - t_round

        rows.append([
            str(r), sign, f"{axis_shift(chosen_dlogit):+.3f}",
            *(f"{chosen_dlogit[f]['mean']:+.2f}" for f in FOUNDATION_ORDER),
            f"{round_pmass:.3f}", f"{round_ppl:.2f}", f"{elapsed:.0f}",
        ])

        (args.out / f"round_{r:02d}.json").write_text(json.dumps({
            "round": r, "sign": sign, "calibrated_C": C, "signed_C": signed_C,
            "kl_p95_at_calib": kl_hit,
            "dlogit_per_foundation": chosen_dlogit,
            "axis_shift": axis_shift(chosen_dlogit),
            "auth_logit_pos": auth_pos, "auth_logit_neg": auth_neg,
            "pmass_mean": round_pmass, "ppl": round_ppl,
            "raw_p_true": chosen_report["raw"],
            "raw_pmass": chosen_report["raw_pmass"],
            "raw_nll": chosen_report.get("raw_nll", {}),
            "elapsed_s": elapsed,
        }, indent=2))

        round_summaries.append({
            "round": r, "sign": sign, "signed_C": signed_C,
            "auth_dlogit_mean": _auth_logit(chosen_dlogit),
            "axis_shift": axis_shift(chosen_dlogit),
            "pmass": round_pmass, "ppl": round_ppl, "elapsed_s": elapsed,
        })

    # Save the accumulated vector.
    if v_running is not None:
        v_running.save(str(args.out / "v_accum.safetensors"))

    headers = (["r", "±", "axis"]
               + [FOUNDATION_SHORT[f] for f in FOUNDATION_ORDER]
               + ["pmass", "ppl", "t_s"])
    tsv = tabulate(rows, headers=headers, tablefmt="tsv")
    (args.out / "rounds.tsv").write_text(tsv)

    (args.out / "meta.json").write_text(json.dumps({
        "meta": meta, "args": vars(args), "layers": list(layers),
        "rounds": round_summaries,
        "bare": {
            "pmass": base_pmass, "ppl": base_ppl,
            "logit_per_foundation": base_logit_per_f,
        },
    }, indent=2, default=str))
    append_run(args.out, {**meta, "kind": "iterated_steer", "rounds": round_summaries})

    logger.info(f"\nout: {args.out}")
    logger.info(f"v_accum: {args.out / 'v_accum.safetensors'}")
    logger.info("\n" + tsv)


def _eval_with(v_running, v_signed_fresh, model, tok, args, base_report):
    """Eval at v_running + v_signed_fresh. Both must be coeff-baked (coeff=1)
    so their sum is meaningful. Returns (dlogit, report)."""
    v_combined = v_signed_fresh if v_running is None else v_running + v_signed_fresh
    with v_combined(model):
        report = evaluate_with_vector(model, tok, name=args.vignettes,
                                      max_think_tokens=args.max_think_tokens, vector=v_combined)
    return dlogit_per_foundation(base_report, report, args.vignettes), report


if __name__ == "__main__":
    main()
