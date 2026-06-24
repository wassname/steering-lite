"""All-instrument tinymfv showcase: ONE calibrated activation-steering vector,
administered across every tinymfv instrument, 3-point (base / +C / -C).

This is the dogfooding run for tinymfv's plotting before the lib is published:
the MFV method-comparison sweep (run_tinymfv_sweep.py) only touches the nominal
vignette eval. Here we take a single interpretable vector (default mean_diff,
the Authority/Care persona axis) and run it through BOTH tinymfv eval paths:

  - ordinal surveys (mfq2, big5, 16pf, humor_styles) via `tinymfv.administer`
    -> per-factor profile on the 1-5/1-7 scale, with a coherence pmass check.
  - nominal MFV vignettes (mfv == "classic") via `evaluate_multibool`
    -> per-foundation Delta-logit vs bare.

We extract + iso-KL calibrate the vector ONCE, then for each instrument
administer at coeff 0 (bare), +C, -C. Output is per-instrument profile CSVs
(`<instr>_c{base,pos,neg}_foundations.csv`, foundation/mean/pmass) + an
`mfv.json`, consumed by tinymfv's plotting (scripts/plot_steer_showcase.py in
the tinymfv repo) to draw the range/map/foundation figures.

  uv run --extra benchmark python scripts/run_allinstr_showcase.py \
    --model Qwen/Qwen3-4B --out outputs/allinstr_qwen3_4b
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite._quiet import quiet_external_logs
from _meta import make_metadata, append_run
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY
from steering_lite.eval.tinymfv import evaluate_multibool
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, baseline_logit_per_foundation, dlogit_per_foundation, format_cell,
)

from tinymfv import get_instrument, administer

quiet_external_logs()
logger.remove()
logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=False, format="{message}")

ORDINAL_INSTRUMENTS = ["mfq2", "big5", "16pf", "humor_styles"]


def _make_cfg(method: str, layers: tuple[int, ...]) -> sl.SteeringConfig:
    common = dict(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0)
    table = {
        "mean_diff": sl.MeanDiffC(**common),
        "pca": sl.PCAC(**common),
        "sspace": sl.SSpaceC(**common, r=-1),
    }
    return table[method]


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


def _administer_profile(model, tok, instr, *, batch_size: int, max_think_tokens: int) -> dict:
    """administer once -> {foundation: (mean, pmass)} aligned to instr.dimensions.
    max_think_tokens > 0 so the steer accrues over the think trace before the answer slot."""
    res = administer(model, tok, instr, batch_size=batch_size, max_think_tokens=max_think_tokens)
    return {f["foundation"]: (float(f["mean"]), float(res["mean_pmass_allowed"]))
            for f in res["foundations"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--method", default="mean_diff", choices=["mean_diff", "pca", "sspace"])
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--admin-batch-size", type=int, default=36)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--target-kl", type=float, default=0.5)
    ap.add_argument("--calib-T", type=int, default=60)
    ap.add_argument("--calib-iters", type=int, default=9)
    ap.add_argument("--max-think-tokens", type=int, default=256)  # MFV vignette think budget
    ap.add_argument("--admin-think-tokens", type=int, default=64)  # ordinal survey think budget (spec "light")
    ap.add_argument("--fixed-C", type=float, default=None,
                    help="skip iso-KL calibration and deploy this coefficient (iso-KL 0.5 gave a too-gentle 0.38)")
    ap.add_argument("--instruments", nargs="*", default=ORDINAL_INSTRUMENTS + ["mfv"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%dT%H%M%S")
        model_short = args.model.split("/")[-1].lower().replace("-", "_").replace(".", "")
        args.out = Path(f"outputs/{ts}_allinstr_{model_short}")
    args.out.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(args)
    logger.info(f"run_id={meta['run_id']} commit={meta['git_commit']} ts={meta['timestamp']}")
    logger.info(f"BLUF: model={args.model} method={args.method} instruments={args.instruments} "
                f"target_kl={args.target_kl}")

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    # === one vector, extracted + iso-KL calibrated once ===================
    pos_prompts, neg_prompts = make_persona_pairs(
        tok, n_pairs=args.n_pairs, thinking=True, persona_pairs=PERSONA_PAIRS_AUTHORITY)
    calib_prompts = _calib_prompts(tok, n=8)
    cfg = _make_cfg(args.method, layers)
    logger.info(f"\n=== train+calibrate steer_{args.method} ===")
    t0 = time.time()
    v = sl.train(model, tok, pos_prompts, neg_prompts, cfg,
                 batch_size=args.batch_size, max_length=args.max_length)
    if args.fixed_C is not None:
        C, kl_hit = float(args.fixed_C), float("nan")
        logger.info(f"fixed C={C:+.4f} (iso-KL skipped) elapsed={time.time()-t0:.0f}s")
    else:
        coeff_calib, _hist = sl.calibrate_iso_kl(
            v, model, tok, calib_prompts, target_kl=args.target_kl, T=args.calib_T,
            max_iters=args.calib_iters, device=args.device, bracket=(0.01, 1e6))
        C = float(coeff_calib)
        kl_hit = _hist[-1].get("kl_p95", float("nan")) if _hist else float("nan")
        logger.info(f"calibrated C={C:+.4f} kl_p95={kl_hit:.3f} elapsed={time.time()-t0:.0f}s")

    summary: dict = {"meta": meta, "model": args.model, "method": args.method,
                     "layers": list(layers), "calibrated_C": C, "kl_p95_at_calib": kl_hit,
                     "instruments": {}}

    # === ordinal instruments: administer at base / +C / -C ================
    poles = [("base", None), ("pos", +C), ("neg", -C)]
    for name in [n for n in args.instruments if n in ORDINAL_INSTRUMENTS]:
        instr = get_instrument(name)
        logger.info(f"\n=== administer {name} ({instr.display}, {len(instr.dimensions)} factors) ===")
        prof_by_pole = {}
        for tag, coeff in poles:
            if coeff is None:
                prof_by_pole[tag] = _administer_profile(model, tok, instr, batch_size=args.admin_batch_size,
                                                        max_think_tokens=args.admin_think_tokens)
            else:
                with v(model, C=coeff):
                    prof_by_pole[tag] = _administer_profile(model, tok, instr, batch_size=args.admin_batch_size,
                                                            max_think_tokens=args.admin_think_tokens)
            pm = list(prof_by_pole[tag].values())[0][1]
            logger.info(f"  {name} {tag} (C={coeff}): pmass={pm:.3f} "
                        f"profile=" + ", ".join(f"{f}={prof_by_pole[tag][f][0]:.2f}" for f in instr.dimensions))
        rows = [{"foundation": f, "pole": tag, "c": {"base": 0, "pos": 1, "neg": -1}[tag],
                 "mean": prof_by_pole[tag][f][0], "pmass": prof_by_pole[tag][f][1]}
                for tag, _ in poles for f in instr.dimensions]
        with open(args.out / f"{name}_profiles.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["foundation", "pole", "c", "mean", "pmass"])
            w.writeheader()
            w.writerows(rows)
        summary["instruments"][name] = {"display": instr.display, "dimensions": instr.dimensions,
                                        "pmass_base": prof_by_pole["base"][instr.dimensions[0]][1]}

    # === nominal MFV: evaluate at base / +C / -C ==========================
    if "mfv" in args.instruments:
        logger.info("\n=== evaluate MFV (classic vignettes) base / +C / -C ===")
        base_report = evaluate_multibool(model, tok, name="classic",
                                         max_think_tokens=args.max_think_tokens, batch_size=args.eval_batch_size)
        base_logit = baseline_logit_per_foundation(base_report)
        with v(model, C=+C):
            pos_report = evaluate_multibool(model, tok, name="classic",
                                            max_think_tokens=args.max_think_tokens, batch_size=args.eval_batch_size)
        with v(model, C=-C):
            neg_report = evaluate_multibool(model, tok, name="classic",
                                            max_think_tokens=args.max_think_tokens, batch_size=args.eval_batch_size)
        pos_dlogit = dlogit_per_foundation(base_report, pos_report)
        neg_dlogit = dlogit_per_foundation(base_report, neg_report)
        logger.info("  MFV base logit: " + ", ".join(f"{f}={format_cell(base_logit[f])}" for f in FOUNDATION_ORDER))
        (args.out / "mfv.json").write_text(json.dumps({
            "base_logit_per_foundation": base_logit,
            "pos": {"coeff": +C, "dlogit_per_foundation": pos_dlogit},
            "neg": {"coeff": -C, "dlogit_per_foundation": neg_dlogit},
            "foundation_order": list(FOUNDATION_ORDER),
        }, indent=2))
        summary["instruments"]["mfv"] = {"display": "MFV vignettes", "foundations": list(FOUNDATION_ORDER)}

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    append_run(args.out, {**meta, "kind": "allinstr", "method": args.method, "calibrated_C": C})
    logger.info(f"\n=== allinstr showcase complete -> {args.out} ===")
    logger.info("SHOULD: one <instr>_profiles.csv per ordinal instrument (base/pos/neg x factors, "
                "pmass near 1.0 = coherent) + mfv.json. ELSE pmass<<1 means steering broke the readout.")


if __name__ == "__main__":
    main()
