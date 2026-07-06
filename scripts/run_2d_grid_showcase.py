#!/usr/bin/env python3
"""2D honesty-c x credulity-c grid administration.

Loads two trained+calibrated sspace vectors (saved via --save-vector), orthogonalizes
their dS directions in S-space, then for each (hc, cc) grid point builds a combined
vector v_h_orth*(C_h*hc) + v_c_orth*(C_c*cc) and administers it across all instruments.

The sspace gate="cosine" path independently gates each stacked row by its cosine to the
activation, so pre-scaling dS by (C_axis * c_mult) bakes the per-axis coefficient in, and
cfg.coeff=1.0 on the combined vector passes both through.

Output: {out}/{instrument}_profiles.csv with columns including honesty_c, credulity_c
(instead of the 1D c), plus a summary.json with the cosine-before-orth and per-axis C.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
from einops import einsum
from jaxtyping import Float
from loguru import logger

import steering_lite as sl
from steering_lite.vector import Vector
from steering_lite.eval.tinymfv import evaluate_multibool
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, baseline_logit_per_foundation, dlogit_per_foundation,
)
from tinymfv import get_instrument
from tinymfv.administer import administer

ORDINAL_INSTRUMENTS = ["mfq2", "big5", "humor_styles"]
# FOUNDATION_ORDER imported from foundations.py: ["Care", "Sanctity", "Authority",
# "Loyalty", "Fairness", "Liberty", "Social Norms"]. Do NOT override with lowercase -
# baseline_logit_per_foundation returns capitalized keys, so a lowercase override
# causes KeyError on the first MFV cell.


def _administer_profile(model, tok, instr, *, batch_size, max_think_tokens,
                        n_samples, temperature, top_p):
    res = administer(model, tok, instr, batch_size=batch_size,
                     max_think_tokens=max_think_tokens,
                     n_samples=n_samples, temperature=temperature, top_p=top_p)
    pm = float(res["mean_pmass_allowed"])
    return {f["foundation"]: {"E": float(f["mean"]), "C": float(f["C"]),
                              "C_sd": float(f["C_sd"]), "E_sd": float(f["sd"]),
                              "E_ci95_lo": float(f["ci95_lo"]),
                              "E_ci95_hi": float(f["ci95_hi"]),
                              "C_ci95_lo": float(f["C_ci95_lo"]),
                              "C_ci95_hi": float(f["C_ci95_hi"]),
                              "framing_spread": float(f["framing_spread"]),
                              "logodds": float(f["logodds_agree"]), "pmass": pm}
            for f in res["foundations"]}


def orthogonalize_ds(v_a: Vector, v_b: Vector) -> tuple[Vector, Vector, float]:
    """Project each vector's dS out of the other, in S-space. Returns (v_a_orth, v_b_orth, cosine_before).

    For sspace, stacked[layer_key]["dS"] has shape [k, r]. Single-axis vectors have k=1.
    We Gram-Schmidt both directions so they are mutually orthogonal in every layer's S-space.
    """
    cosines = []
    new_stacked_a, new_stacked_b = {}, {}
    for li in v_a.stacked:
        dS_a = v_a.stacked[li]["dS"].float()   # [k_a, r]
        dS_b = v_b.stacked[li]["dS"].float()   # [k_b, r]
        # cosine between the two direction sets (use mean over k if k>1)
        # for k=1 this is just the single direction cosine
        a_flat = dS_a.mean(dim=0)   # [r]
        b_flat = dS_b.mean(dim=0)   # [r]
        cos = float(torch.dot(a_flat, b_flat) /
                    (a_flat.norm() * b_flat.norm() + 1e-12))
        cosines.append(cos)
        # Gram-Schmidt: a_orth = a - proj_b(a), b_orth = b - proj_a(b)
        proj_b_a = (torch.dot(a_flat, b_flat) /
                    (torch.dot(b_flat, b_flat) + 1e-12)) * b_flat
        proj_a_b = (torch.dot(b_flat, a_flat) /
                    (torch.dot(a_flat, a_flat) + 1e-12)) * a_flat
        dS_a_orth = dS_a - proj_b_a.unsqueeze(0)
        dS_b_orth = dS_b - proj_a_b.unsqueeze(0)
        new_stacked_a[li] = {**v_a.stacked[li], "dS": dS_a_orth.to(v_a.stacked[li]["dS"].dtype)}
        new_stacked_b[li] = {**v_b.stacked[li], "dS": dS_b_orth.to(v_b.stacked[li]["dS"].dtype)}
    from copy import deepcopy
    v_a_orth = Vector(deepcopy(v_a.cfg), v_a.shared, new_stacked_a)
    v_b_orth = Vector(deepcopy(v_b.cfg), v_b.shared, new_stacked_b)
    mean_cos = sum(cosines) / len(cosines) if cosines else 0.0
    return v_a_orth, v_b_orth, mean_cos


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--vector-a", type=Path, required=True, help="honesty vector .safetensors")
    ap.add_argument("--vector-b", type=Path, required=True, help="credulity vector .safetensors")
    ap.add_argument("--summary-a", type=Path, required=True, help="honesty summary.json (for calibrated C)")
    ap.add_argument("--summary-b", type=Path, required=True, help="credulity summary.json (for calibrated C)")
    ap.add_argument("--label-a", default="honesty")
    ap.add_argument("--label-b", default="credulity")
    ap.add_argument("--hc-grid", default="-1,-0.5,0,0.5,1",
                    help="honesty-c multipliers of calibrated C")
    ap.add_argument("--cc-grid", default="-1,-0.5,0,0.5,1",
                    help="credulity-c multipliers of calibrated C")
    ap.add_argument("--instruments", nargs="*", default=ORDINAL_INSTRUMENTS + ["mfv"])
    ap.add_argument("--admin-batch-size", type=int, default=4)
    ap.add_argument("--admin-n-samples", type=int, default=8)
    ap.add_argument("--admin-temperature", type=float, default=0.7)
    ap.add_argument("--admin-top-p", type=float, default=0.95)
    ap.add_argument("--admin-think-tokens", type=int, default=64)
    ap.add_argument("--max-think-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # load vectors + calibrated C
    v_a = Vector.load(str(args.vector_a))
    v_b = Vector.load(str(args.vector_b))
    C_a = float(json.loads(args.summary_a.read_text())["calibrated_C"])
    C_b = float(json.loads(args.summary_b.read_text())["calibrated_C"])
    logger.info(f"loaded {args.label_a} C={C_a:.4f} k={v_a.k_rounds} and "
                f"{args.label_b} C={C_b:.4f} k={v_b.k_rounds}")

    # orthogonalize
    v_a_orth, v_b_orth, cos_before = orthogonalize_ds(v_a, v_b)
    logger.info(f"cosine between {args.label_a} and {args.label_b} dS (before orth): {cos_before:.4f}")
    if abs(cos_before) > 0.95:
        logger.warning(f"axes near-collinear (cos={cos_before:.4f}); 2D grid may have no spread. "
                       f"Falling back to raw add (no orthogonalization) would not help here.")

    # load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, args.torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(args.model)

    hc_grid = [float(x) for x in args.hc_grid.split(",")]
    cc_grid = [float(x) for x in args.cc_grid.split(",")]

    summary = {
        "model": args.model, "method": v_a.cfg.method,
        "label_a": args.label_a, "label_b": args.label_b,
        "C_a": C_a, "C_b": C_b,
        "cosine_before_orth": cos_before,
        "hc_grid": hc_grid, "cc_grid": cc_grid,
        "instruments": {},
    }

    # === ordinal instruments ===
    for name in [n for n in args.instruments if n in ORDINAL_INSTRUMENTS]:
        instr = get_instrument(name)
        logger.info(f"\n=== administer {name} over {len(hc_grid)}x{len(cc_grid)} grid ===")
        rows = []
        for hc in hc_grid:
            for cc in cc_grid:
                # build combined vector: pre-scale dS by (C_axis * c_mult), then stack
                v_h = v_a_orth * (C_a * hc) if hc != 0 else None
                v_c = v_b_orth * (C_b * cc) if cc != 0 else None
                if v_h is not None and v_c is not None:
                    v_comb = v_h + v_c
                    with v_comb(model, C=1.0):
                        prof = _administer_profile(model, tok, instr,
                                                   batch_size=args.admin_batch_size,
                                                   max_think_tokens=args.admin_think_tokens,
                                                   n_samples=args.admin_n_samples,
                                                   temperature=args.admin_temperature,
                                                   top_p=args.admin_top_p)
                elif v_h is not None:
                    with v_h(model, C=1.0):
                        prof = _administer_profile(model, tok, instr,
                                                   batch_size=args.admin_batch_size,
                                                   max_think_tokens=args.admin_think_tokens,
                                                   n_samples=args.admin_n_samples,
                                                   temperature=args.admin_temperature,
                                                   top_p=args.admin_top_p)
                elif v_c is not None:
                    with v_c(model, C=1.0):
                        prof = _administer_profile(model, tok, instr,
                                                   batch_size=args.admin_batch_size,
                                                   max_think_tokens=args.admin_think_tokens,
                                                   n_samples=args.admin_n_samples,
                                                   temperature=args.admin_temperature,
                                                   top_p=args.admin_top_p)
                else:
                    prof = _administer_profile(model, tok, instr,
                                               batch_size=args.admin_batch_size,
                                               max_think_tokens=args.admin_think_tokens,
                                               n_samples=args.admin_n_samples,
                                               temperature=args.admin_temperature,
                                               top_p=args.admin_top_p)
                pm = list(prof.values())[0]["pmass"]
                logger.info(f"  {name} hc={hc:+.1f} cc={cc:+.1f}: pmass={pm:.3f} "
                            + ", ".join(f"{f}={prof[f]['C']:+.2f}" for f in instr.dimensions))
                for f in instr.dimensions:
                    rows.append({"foundation": f, "honesty_c": hc, "credulity_c": cc,
                                 "mean": prof[f]["E"], "C": prof[f]["C"],
                                 "E_sd": prof[f]["E_sd"], "E_ci95_lo": prof[f]["E_ci95_lo"],
                                 "E_ci95_hi": prof[f]["E_ci95_hi"],
                                 "C_sd": prof[f]["C_sd"], "C_ci95_lo": prof[f]["C_ci95_lo"],
                                 "C_ci95_hi": prof[f]["C_ci95_hi"],
                                 "framing_spread": prof[f]["framing_spread"],
                                 "logodds": prof[f]["logodds"], "pmass": prof[f]["pmass"]})
        with open(args.out / f"{name}_profiles.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["foundation", "honesty_c", "credulity_c",
                                               "mean", "E_sd", "E_ci95_lo", "E_ci95_hi",
                                               "C", "C_sd", "C_ci95_lo", "C_ci95_hi",
                                               "framing_spread", "logodds", "pmass"])
            w.writeheader()
            w.writerows(rows)
        summary["instruments"][name] = {"display": instr.display,
                                        "dimensions": instr.dimensions,
                                        "pmass_base": next(r["pmass"] for r in rows
                                                           if r["honesty_c"] == 0 and r["credulity_c"] == 0)}

    # === MFV ===
    if "mfv" in args.instruments:
        logger.info(f"\n=== administer mfv over {len(hc_grid)}x{len(cc_grid)} grid ===")
        mfv_kw = dict(name="classic", log_demo=False, verbose=0,
                      max_think_tokens=args.max_think_tokens, batch_size=args.admin_batch_size)
        def coh(rep):
            i = rep["info"]
            return {"mean_margin": rep["mean_margin"], "frac_unscorable": i["frac_unscorable"],
                    "mean_pmass_allowed": i["mean_pmass_allowed"], "mean_nll_prefill": i["mean_nll_prefill"]}
        # base report (hc=0, cc=0)
        base_report = evaluate_multibool(model, tok, **mfv_kw)
        base_logit = baseline_logit_per_foundation(base_report)
        rows = []
        for hc in hc_grid:
            for cc in cc_grid:
                v_h = v_a_orth * (C_a * hc) if hc != 0 else None
                v_c = v_b_orth * (C_b * cc) if cc != 0 else None
                if v_h is not None and v_c is not None:
                    v_comb = v_h + v_c
                    ctx = v_comb(model, C=1.0)
                elif v_h is not None:
                    ctx = v_h(model, C=1.0)
                elif v_c is not None:
                    ctx = v_c(model, C=1.0)
                else:
                    ctx = None
                if ctx is not None:
                    with ctx:
                        rep = evaluate_multibool(model, tok, **mfv_kw)
                else:
                    rep = base_report
                cinfo = coh(rep)
                dl = (dlogit_per_foundation(base_report, rep) if ctx is not None
                      else {f: {"mean": 0.0, "std": 0.0, "sem": 0.0} for f in FOUNDATION_ORDER})
                logger.info(f"  mfv hc={hc:+.1f} cc={cc:+.1f}: margin={cinfo['mean_margin']:+.2f}nat "
                            f"unscorable={cinfo['frac_unscorable']:.3f} pmass={cinfo['mean_pmass_allowed']:.3f}")
                for f in FOUNDATION_ORDER:
                    rows.append({"foundation": f, "honesty_c": hc, "credulity_c": cc,
                                 "mean": base_logit[f]["mean"] + dl[f]["mean"],
                                 "dlogit": dl[f]["mean"], "dlogit_sd": dl[f]["std"],
                                 "dlogit_sem": dl[f]["sem"],
                                 "pmass": cinfo["mean_pmass_allowed"],
                                 "mean_margin": cinfo["mean_margin"],
                                 "frac_unscorable": cinfo["frac_unscorable"],
                                 "mean_nll_prefill": cinfo["mean_nll_prefill"]})
        with open(args.out / "mfv_profiles.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["foundation", "honesty_c", "credulity_c",
                                               "mean", "dlogit", "dlogit_sd", "dlogit_sem",
                                               "pmass", "mean_margin", "frac_unscorable",
                                               "mean_nll_prefill"])
            w.writeheader()
            w.writerows(rows)
        summary["instruments"]["mfv"] = {"display": "MFV vignettes",
                                         "foundations": list(FOUNDATION_ORDER)}

    # save combined vector at the (+1, +1) corner for reproducibility
    v_comb_corner = (v_a_orth * C_a) + (v_b_orth * C_b)
    v_comb_corner.save(str(args.out / "combined_vector.safetensors"))
    logger.info(f"saved combined vector to {args.out / 'combined_vector.safetensors'}")

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"\n=== 2D grid done; wrote {len(summary['instruments'])} instruments to {args.out} ===")


if __name__ == "__main__":
    main()
