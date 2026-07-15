"""Gate before the full 11-method rerun: which foundation-target persona moves its OWN foundation
cleanly (on MFQ-2 and MFV) without dragging the others? The showcase Care/Authority axis half-fails
at -C (collapses to "Social Norms = not wrong"). The model saturates high on Care/Authority, so we
try personas that steer toward NON-saturated directions (+purity, -authority, +loyalty, +equality,
traditionalism), built from the persona-steering-template-library's rules (short mirrored enactable
poles) + its top template ("Use the priorities of a {persona} person.").

Per persona: train ONE mean_diff vector (the headline method), then at +C / -C report the per-
foundation shift on MFQ-2 (1-5 endorsement) and MFV (dclr), plus pmass coherence. Read by eye
(bayesian): keep the persona whose named foundation moves and whose off-target foundations don't.
Only the winners go to the full method sweep + all-instrument showcase + README.

  uv run --extra benchmark python scripts/probe_foundation_selectivity.py --model Qwen/Qwen3-4B
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.data import make_persona_pairs, PERSONA_REGISTRY
from steering_lite.eval.tinymfv import evaluate_multibool
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, baseline_clr_per_foundation, dclr_per_foundation,
)
from moralmaps import get_instrument, administer


def _mfq2(model, tok, instr, bs, think):
    res = administer(model, tok, instr, batch_size=bs, max_think_tokens=think)
    return (np.array([f["mean"] for f in res["foundations"]], float),
            float(res["mean_pmass_allowed"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--personas", nargs="*",
                    default=["authority_care", "traditionalist", "sanctity", "authority", "loyalty", "equality"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--admin-batch-size", type=int, default=36)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--admin-think", type=int, default=64)
    ap.add_argument("--mfv-think", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/foundation_probe"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(args.device).eval()
    n = model.config.num_hidden_layers
    layers = tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))

    instr = get_instrument("mfq2")
    mfq2_dims = instr.dimensions
    mfv_kw = dict(name="classic", log_demo=False, verbose=0, max_think_tokens=args.mfv_think,
                  batch_size=args.eval_batch_size)
    base_prof, base_pm = _mfq2(model, tok, instr, args.admin_batch_size, args.admin_think)
    base_mfv = evaluate_multibool(model, tok, **mfv_kw)
    base_clr = baseline_clr_per_foundation(base_mfv)
    logger.info(f"BASE mfq2 pmass={base_pm:.3f} | mfq2={dict(zip(mfq2_dims, base_prof.round(2)))}")

    results = {"model": args.model, "C": args.C, "base_mfq2": dict(zip(mfq2_dims, base_prof.tolist())),
               "base_mfv_clr": {f: base_clr[f]["mean"] for f in FOUNDATION_ORDER}, "personas": {}}
    logger.info("SHOULD: each persona's NAMED foundation moves most; off-target small; pmass>=0.9 both poles.")
    for pname in args.personas:
        pairs, template = PERSONA_REGISTRY[pname]
        pos, neg = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True,
                                      persona_pairs=pairs, template=template)
        t0 = time.time()
        v = sl.train(model, tok, pos, neg,
                     sl.MeanDiffC(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0),
                     batch_size=8, max_length=384)
        cell = {"pairs": pairs, "template": template}
        for sign, C in [("pos", +args.C), ("neg", -args.C)]:
            with v(model, C=C):
                prof, pm = _mfq2(model, tok, instr, args.admin_batch_size, args.admin_think)
                mfv = evaluate_multibool(model, tok, **mfv_kw)
            dl = dclr_per_foundation(base_mfv, mfv)
            cell[sign] = {
                "mfq2_delta": dict(zip(mfq2_dims, (prof - base_prof).round(3).tolist())),
                "mfq2_pmass": round(pm, 3),
                "mfv_dclr": {f: round(dl[f]["mean"], 3) for f in FOUNDATION_ORDER},
                "mfv_pmass": round(float(mfv["info"]["mean_pmass_allowed"]), 3),
            }
        results["personas"][pname] = cell
        mq = "  ".join(f"{d}{cell['pos']['mfq2_delta'][d]:+.2f}/{cell['neg']['mfq2_delta'][d]:+.2f}" for d in mfq2_dims)
        mv = "  ".join(f"{f.split()[0]}{cell['pos']['mfv_dclr'][f]:+.2f}/{cell['neg']['mfv_dclr'][f]:+.2f}" for f in FOUNDATION_ORDER)
        logger.info(f"\n=== {pname} ({pairs[0][0]} / {pairs[0][1]}) {time.time()-t0:.0f}s "
                    f"pmass +{cell['pos']['mfq2_pmass']}/-{cell['neg']['mfq2_pmass']} ===")
        logger.info(f"  mfq2 d(+C/-C): {mq}")
        logger.info(f"  MFV  d(+C/-C): {mv}")
        (args.out / "selectivity.json").write_text(json.dumps(results, indent=2))
    logger.info(f"\n=== probe complete -> {args.out / 'selectivity.json'} ===")


if __name__ == "__main__":
    main()
