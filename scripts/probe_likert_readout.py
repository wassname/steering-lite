"""Does the bounded-expectation Likert readout (E in [1,5]) wash out steer direction that an
unbounded log-odds readout keeps? Hypothesis: steering raises the entropy of the within-5-token
answer distribution p, so E -> 3 (mid-scale) for every foundation regardless of direction, while
pmass (mass on the 5 digits) stays ~1.0 and hides it. MFV is clean only because dlogit is unbounded.

We reuse the PRODUCTION readout (read_items + per_item_categorical -> frame-averaged forward p per
item), then score each item three ways and pool per foundation (reverse-keyed):

  E        = sum_k k * p[k]                       bounded in [1, scale_max], midpoint = (1+M)/2
  mode     = argmax_k p[k] + 1                    bounded, discrete (tail-insensitive)
  logodds  = log( (p4+p5) / (p1+p2) )             unbounded, signed (the MFV-style readout)
  entropy  = -sum_k p[k] log p[k]                 coherence-within-allowed (the gate pmass misses)

Run at base / +2C / -2C of the calibrated equality (sspace) vector on mfq2 only (think=64, ~10 min).

  uv run --extra benchmark python scripts/probe_likert_readout.py --model Qwen/Qwen3-4B --C 35.27
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tabulate import tabulate
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.data import make_persona_pairs, PERSONA_REGISTRY
from tinymfv import get_instrument
from tinymfv.read import read_items, resolve_answer_ids
from tinymfv.instrument import per_item_categorical


def score_profile(model, tok, instr, *, batch_size: int, max_think_tokens: int) -> dict:
    """Per-foundation pooled, reverse-keyed readouts from the SAME frame-averaged p per item."""
    answer_ids = resolve_answer_ids(tok, instr.answer_space)
    per_row = read_items(model, tok, instr, instr.items, answer_ids,
                         max_think_tokens=max_think_tokens, batch_size=batch_size)
    items = per_item_categorical(per_row, instr.kind)            # {id: {p(forward, frame-avg), dimension, sign}}
    M = instr.scale_max
    w = np.arange(1, M + 1, dtype=float)
    agree, disagree = [M - 1, M - 2], [0, 1]                     # top-2 vs bottom-2 indices (0-based)
    by_found: dict[str, dict[str, list]] = {}
    for it in items.values():
        p = np.asarray(it["p"], float)
        E = float((p * w).sum())
        mode = float(np.argmax(p) + 1)
        lo = float(np.log((p[agree].sum() + 1e-9) / (p[disagree].sum() + 1e-9)))
        ent = float(-(p * np.log(p + 1e-12)).sum())
        keyed = lambda x, mid: (2 * mid - x) if it["sign"] < 0 else x   # reverse-key around the midpoint
        d = by_found.setdefault(it["dimension"], {"E": [], "mode": [], "lo": [], "ent": []})
        d["E"].append(keyed(E, (1 + M) / 2))
        d["mode"].append(keyed(mode, (1 + M) / 2))
        d["lo"].append(keyed(lo, 0.0))                                  # log-odds keyed around 0
        d["ent"].append(ent)
    return {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in by_found.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--persona", default="equality")
    ap.add_argument("--C", type=float, default=35.27, help="calibrated coefficient from the showcase run")
    ap.add_argument("--mults", default="0,2,-2", help="signed multipliers of C to administer")
    ap.add_argument("--layers-frac", default="0.2,0.8")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--admin-batch-size", type=int, default=36)
    ap.add_argument("--admin-think-list", default="64,256",
                    help="think budgets to administer at. 64 = the MFQ-2 default; 256 = the MFV budget, "
                         "to control the think-budget confound the external review flagged (MFV got 256).")
    ap.add_argument("--out", type=Path, default=Path("outputs/likert_readout_probe"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    n = model.config.num_hidden_layers
    lo_f, hi_f = (float(x) for x in args.layers_frac.split(","))
    layers = tuple(range(max(2, int(n * lo_f)), min(n - 2, int(n * hi_f))))

    pairs, template = PERSONA_REGISTRY[args.persona]
    pos, neg = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True, persona_pairs=pairs, template=template)
    v = sl.train(model, tok, pos, neg, sl.SSpaceC(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0, r=-1),
                 batch_size=8, max_length=384)
    logger.info(f"trained sspace {args.persona} ({pairs[0][0]} vs {pairs[0][1]}) on layers {layers}")

    instr = get_instrument("mfq2")
    mults = [float(m) for m in args.mults.split(",")]
    thinks = [int(t) for t in args.admin_think_list.split(",")]

    # {think: {mult: {foundation: {E,mode,lo,ent}}}} -- saved to disk so nothing depends on the
    # truncated stdout buffer (pueue keeps only the tail; the per-c tables scrolled off last time).
    allprofs: dict = {}
    for think in thinks:
        profs: dict[float, dict] = {}
        for m in mults:
            coeff = m * args.C
            if m == 0:
                profs[m] = score_profile(model, tok, instr, batch_size=args.admin_batch_size, max_think_tokens=think)
            else:
                with v(model, C=coeff):
                    profs[m] = score_profile(model, tok, instr, batch_size=args.admin_batch_size, max_think_tokens=think)
            logger.info(f"administered mfq2 at think={think} c={m:+g} (coeff={coeff:+.2f})")
        allprofs[think] = profs
        (args.out / "readouts.json").write_text(json.dumps(
            {str(t): {str(m): p for m, p in pr.items()} for t, pr in allprofs.items()}, indent=2))

        print(f"\n############### THINK BUDGET = {think} tokens ###############")
        for key, label in [("E", "EXPECTATION E (bounded 1-5, midpoint 3)"),
                           ("mode", "ARGMAX/MODE (bounded)"),
                           ("lo", "LOG-ODDS log[(p4+p5)/(p1+p2)] (unbounded, signed; reverse-keyed around 0)"),
                           ("ent", "ENTROPY of p, nats (coherence the pmass gate misses; uniform=1.609)")]:
            rows = [[f] + [round(profs[m][f][key], 3) for m in mults] for f in instr.dimensions]
            print(f"\n=== {label} ===")
            print(tabulate(rows, headers=["foundation"] + [f"c={m:+g}" for m in mults],
                           tablefmt="pipe", floatfmt="+.3f"))

        # odd/even decomposition at EVERY |c| with both poles (|c|=1 stays coherent; |c|=2 collapses
        # some foundations to NaN at think=256). A CLEAN bipolar axis is odd-symmetric (directional >>
        # common-mode). GPT-5.5's catch: if MFQ direction~0 and common-mode dominates on log-odds too
        # (not just bounded E), the categorical itself lacks direction -- a response-style shift, not a
        # bounded-readout artifact. nan-safe: skip a foundation whose pole is NaN (collapsed).
        for mm in sorted({abs(m) for m in mults if m > 0}):
            if mm not in profs or -mm not in profs:
                continue
            print(f"\n=== ODD/EVEN at |c|={mm:g} (directional = (d+ - d-)/2, common-mode = (d+ + d-)/2) ===")
            dec = []
            for f in instr.dimensions:
                for key in ("E", "lo"):
                    b, vp, vn = profs[0.0][f][key], profs[mm][f][key], profs[-mm][f][key]
                    if not (np.isfinite(vp) and np.isfinite(vn) and np.isfinite(b)):
                        continue
                    dp, dn = vp - b, vn - b
                    dec.append([f, key, round(b, 3), round(dp, 3), round(dn, 3), round((dp - dn) / 2, 3), round((dp + dn) / 2, 3)])
            print(tabulate(dec, headers=["foundation", "readout", "base", "d(+c)", "d(-c)", "directional", "common-mode"],
                           tablefmt="pipe", floatfmt="+.3f"))

    print("\nSHOULD: on a CLEAN readout the steered axis (equality) is ODD-symmetric -- directional >> common-mode. "
          "If E shows ~0 directional + large common-mode (both poles up) BUT log-odds shows real directional, the "
          "bounded expectation hid it. If BOTH stay common-mode-dominated, the steer truly shifts response-style "
          "(not a moral axis) on MFQ-2 -- a real effect, not a readout artifact. The think=64 vs 256 panels "
          "control the budget confound (MFV ran at 256).")
    logger.info(f"probe done -> {args.out}")


if __name__ == "__main__":
    main()
