"""Goal clause "sweep for the largest coherent C": the showcase used a fixed C=1 by
fiat. This sweeps C and reports, per pole, the coherence (pmass) and the steer
magnitude on the two headline instruments (mfq2 ordinal, MFV nominal), so we can
pick the LARGEST C that still reads coherently at BOTH poles.

  mfq2: administer base/+C/-C -> pmass + mean|profile delta| (1-5 scale)
  MFV : evaluate_multibool +C/-C -> pmass + mean|dlogit| (nats)

"Coherent" = pmass stays high (>= ~0.9) at both poles AND the profile is not pinned
to the neutral midpoint (the degeneracy seen on side instruments at strong -C).

  uv run --extra benchmark python scripts/csweep_coherent_c.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY
from steering_lite.eval.tinymfv import evaluate_multibool
from steering_lite.eval.foundations import FOUNDATION_ORDER, dlogit_per_foundation
from tinymfv import get_instrument, administer


def _mfq2(model, tok, instr, bs):
    res = administer(model, tok, instr, batch_size=bs, max_think_tokens=64)
    prof = np.array([f["mean"] for f in res["foundations"]], dtype=float)
    return prof, float(res["mean_pmass_allowed"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--cs", type=float, nargs="*", default=[0.4, 0.6, 0.8, 1.0, 1.5])
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--admin-batch-size", type=int, default=24)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--skip-mfv", action="store_true", help="ordinal-only sweep (fast); MFV at think=256 is the slow part")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    n = model.config.num_hidden_layers
    layers = tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))
    pos, neg = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True,
                                  persona_pairs=PERSONA_PAIRS_AUTHORITY)
    v = sl.train(model, tok, pos, neg,
                 sl.MeanDiffC(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0),
                 batch_size=8, max_length=384)

    instr = get_instrument("mfq2")
    base_prof, base_pm = _mfq2(model, tok, instr, args.admin_batch_size)
    base_report = None if args.skip_mfv else evaluate_multibool(
        model, tok, name="classic", log_demo=False, verbose=0,
        max_think_tokens=256, batch_size=args.eval_batch_size)
    logger.info(f"base mfq2 pmass={base_pm:.3f}")
    logger.info("  C   | mfq2 pmass+/-  mfq2|d| | MFV pmass+/-  MFV|dlogit|  (pick largest C coherent both poles)")
    for C in args.cs:
        with v(model, C=+C):
            pp, pm_p = _mfq2(model, tok, instr, args.admin_batch_size)
            mfv_p = None if args.skip_mfv else evaluate_multibool(
                model, tok, name="classic", log_demo=False, verbose=0,
                max_think_tokens=256, batch_size=args.eval_batch_size)
        with v(model, C=-C):
            pn, pm_n = _mfq2(model, tok, instr, args.admin_batch_size)
            mfv_n = None if args.skip_mfv else evaluate_multibool(
                model, tok, name="classic", log_demo=False, verbose=0,
                max_think_tokens=256, batch_size=args.eval_batch_size)
        d_ord = float(np.nanmean(np.abs(pp - base_prof)) + np.nanmean(np.abs(pn - base_prof))) / 2
        if args.skip_mfv:
            logger.info(f" {C:+.1f} | {pm_p:.3f}/{pm_n:.3f}  {d_ord:.3f}  | (mfv skipped)")
            continue
        dl_p = dlogit_per_foundation(base_report, mfv_p)
        dl_n = dlogit_per_foundation(base_report, mfv_n)
        d_mfv = float(np.mean([abs(dl_p[f]["mean"]) + abs(dl_n[f]["mean"]) for f in FOUNDATION_ORDER])) / 2
        mfv_pm_p = float(mfv_p["info"]["mean_pmass_allowed"])
        mfv_pm_n = float(mfv_n["info"]["mean_pmass_allowed"])
        logger.info(f" {C:+.1f} | {pm_p:.3f}/{pm_n:.3f}  {d_ord:.3f}  | "
                    f"{mfv_pm_p:.3f}/{mfv_pm_n:.3f}  {d_mfv:.3f}")


if __name__ == "__main__":
    main()
