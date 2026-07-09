"""UAT(2): ordinal steer deltas grow as think tokens go 0 -> 64 -> 512.

The unified ordinal readout generates `think` tokens, the steer accrues over that
trace, THEN we read the prefilled answer slot. So a bigger think budget should let
the same vector move the profile more. This administers mfq2 at base and +C for a
range of think budgets and reports the mean |steered - base| per budget.

think=1 is the floor (HF generate rejects max_new_tokens=0), standing in for "~0".

  uv run --extra benchmark python scripts/ablation_think_budget.py --model Qwen/Qwen3-4B

SHOULD: mean |delta| rises monotonically with the think budget (1 < 64 < 512). A
flat line would mean the steer is read off a single forward and the think budget
does not carry it -- the bug the unification fixed.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY
from moralmaps import get_instrument, administer


def _profile(model, tok, instr, think: int, bs: int) -> tuple[np.ndarray, float]:
    res = administer(model, tok, instr, batch_size=bs, max_think_tokens=think)
    prof = np.array([f["mean"] for f in res["foundations"]], dtype=float)
    return prof, float(res["mean_pmass_allowed"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--instrument", default="mfq2")
    ap.add_argument("--budgets", type=int, nargs="*", default=[1, 64, 512])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--admin-batch-size", type=int, default=36)
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

    instr = get_instrument(args.instrument)
    logger.info(f"SHOULD: mean|delta| rises with think budget on {instr.display}. "
                f"think  mean|delta|  max|delta|  pmass_base  pmass_steer")
    for think in args.budgets:
        base, pm_b = _profile(model, tok, instr, think, args.admin_batch_size)
        with v(model, C=args.C):
            steer, pm_s = _profile(model, tok, instr, think, args.admin_batch_size)
        delta = np.abs(steer - base)
        # nan-safe: a collapsed pole reads NaN ("do not compare"); average the rest
        md = float(np.nanmean(delta))
        mx = float(np.nanmax(delta))
        logger.info(f"{think:>5}  {md:>10.3f}  {mx:>9.3f}  {pm_b:>9.3f}  {pm_s:>10.3f}")


if __name__ == "__main__":
    main()
