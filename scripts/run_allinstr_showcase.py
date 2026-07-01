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
from steering_lite.data import (
    make_persona_pairs,
    make_persona_library_pairs,
    make_moralstory_pairs,
    PERSONA_REGISTRY,
    PERSONA_LIBRARY_TEMPLATE,
)
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
        "sspace_pca": sl.SSpacePCAC(**common, r=-1),
        "corda_pca": sl.CordaPCAC(**common, r=-1),
        "directional_ablation": sl.DirectionalAblationC(**common),
        "linear_act": sl.LinearAcTC(**common),
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


def _read_scenario_prompts(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    prompts = [row["prompt"] if "prompt" in row else row["text"] for row in rows]
    assert prompts, f"no prompts in {path}"
    return prompts


def _stratified_scenario_prompts(path: Path, n: int = 8) -> list[str]:
    prompts = sorted(set(_read_scenario_prompts(path)), key=lambda s: len(s.split()))
    if len(prompts) <= n:
        return prompts
    return [prompts[round(i * (len(prompts) - 1) / (n - 1))] for i in range(n)]


def _persona_library_calib_prompts(tok, scenario_path: Path, n_iid: int = 8, n_ood: int = 4) -> list[str]:
    return _stratified_scenario_prompts(scenario_path, n=n_iid) + _calib_prompts(tok, n=n_ood, seed=1)


def _administer_profile(model, tok, instr, *, batch_size: int, max_think_tokens: int,
                        n_samples: int, temperature: float, top_p: float,
                        sample_path: Path | None = None, pole: str | None = None, c: float | None = None) -> dict:
    """administer once -> {foundation: {E, C, logodds, pmass}} aligned to instr.dimensions.

    E   = expected Likert score (human-comparable, but saturates near a confident answer).
    C   = rank-centered logit contrast sum (k-mid)*logp_k -- the steer-legible readout (sensitive,
          signed, normalizer-invariant); this is what the range/map plots should show for steering.
    logodds = agree-vs-disagree, the readable 2-bin direction.
    max_think_tokens > 0 so the steer accrues over the think trace before the answer slot."""
    res = administer(model, tok, instr, batch_size=batch_size, max_think_tokens=max_think_tokens,
                     n_samples=n_samples, temperature=temperature, top_p=top_p)
    if sample_path is not None:
        assert pole is not None
        assert c is not None
        with sample_path.open("w") as fh:
            for row in res["per_item_frame"]:
                fh.write(json.dumps({
                    "instrument": instr.name,
                    "pole": pole,
                    "c": c,
                    "scale_max": instr.scale_max,
                    "answer_space": instr.answer_space,
                    "id": row["id"],
                    "framing": row["framing"],
                    "foundation": row["foundation"],
                    "sign": row["sign"],
                    "sample_lp": row["sample_lp"],
                    "sample_pmass_allowed": row["sample_pmass_allowed"],
                    "sample_nll_prefill": row["sample_nll_prefill"],
                }) + "\n")
    pm = float(res["mean_pmass_allowed"])
    return {f["foundation"]: {"E": float(f["mean"]), "C": float(f["C"]), "C_sd": float(f["C_sd"]),
                              "E_sd": float(f["sd"]), "E_ci95_lo": float(f["ci95_lo"]),
                              "E_ci95_hi": float(f["ci95_hi"]),
                              "C_ci95_lo": float(f["C_ci95_lo"]), "C_ci95_hi": float(f["C_ci95_hi"]),
                              "framing_spread": float(f["framing_spread"]),
                              "logodds": float(f["logodds_agree"]), "pmass": pm}
            for f in res["foundations"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--method", default="mean_diff",
                    choices=["mean_diff", "pca", "sspace", "sspace_pca", "corda_pca",
                             "directional_ablation", "linear_act"])
    ap.add_argument("--persona", default="authority_care", choices=sorted(PERSONA_REGISTRY),
                    help="axis from PERSONA_REGISTRY (library-template + selectivity-probe validated). "
                         "authority/traditionalist steer NON-saturated directions (model sits near ceiling "
                         "on care+/auth+, so -auth and +trad show the most coherent movement).")
    ap.add_argument("--pairs-source", default="persona", choices=["persona", "persona_library", "moralstory"],
                    help="persona = adjective-prefix contrast; moralstory = foundation-labelled "
                         "situations from moral_stories_foundations (target vs balanced-other); "
                         "persona_library = validated persona-template-library pair + scenario pools.")
    ap.add_argument("--persona-library-dir", type=Path,
                    default=Path("/media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/"
                                 "persona-steering-template-library"))
    ap.add_argument("--persona-library-pair", default="dignity_over_authority")
    ap.add_argument("--persona-library-template", default=PERSONA_LIBRARY_TEMPLATE)
    ap.add_argument("--persona-library-scenarios", type=Path,
                    help="curated scenario JSONL from persona-steering-template-library selection")
    ap.add_argument("--foundation", default="fairness",
                    help="moralstory target foundation (care/fairness/loyalty/authority/sanctity/liberty).")
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=16)
    ap.add_argument("--admin-batch-size", type=int, default=36)
    ap.add_argument("--admin-n-samples", type=int, default=1,
                    help="survey think trajectories per item/frame; >1 requires --admin-temperature > 0")
    ap.add_argument("--admin-temperature", type=float, default=0.0,
                    help="sampling temperature for survey think trajectories")
    ap.add_argument("--admin-top-p", type=float, default=1.0,
                    help="top-p for survey think trajectory sampling")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--target-kl", type=float, default=0.5)
    ap.add_argument("--calib-T", type=int, default=60)
    ap.add_argument("--calib-iters", type=int, default=9)
    ap.add_argument("--max-think-tokens", type=int, default=256)  # MFV vignette think budget
    ap.add_argument("--admin-think-tokens", type=int, default=64)  # ordinal survey think budget (spec "light")
    ap.add_argument("--fixed-C", type=float, default=None,
                    help="skip iso-KL calibration and deploy this coefficient (iso-KL 0.5 gave a too-gentle 0.38)")
    ap.add_argument("--c-grid", default="1",
                    help="comma multipliers of calibrated C for the c-sweep, e.g. '1,2,3'. Each "
                         "evals at +-m*C; '1' reproduces the 3-point base/+C/-C.")
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
    # pairs-source picks the contrast: 'persona' = a one-word adjective prefix (egalitarian vs
    # hierarchical etc.); 'moralstory' = real foundation-labelled SITUATIONS from
    # moral_stories_foundations (target foundation vs balanced-other, situation text only, no
    # completions). The persona "equality" axis conflates fairness with the authority/hierarchy axis
    # (authority moved most, equality barely), so moralstory steers the foundation domain directly.
    # NB moralstory labels MFT foundations; 'fairness' reads on MFQ-2 as equality+proportionality.
    persona_library_meta = None
    if args.pairs_source == "moralstory":
        vec_label = f"{args.method}: {args.foundation} situations vs others"
        logger.info(f"pairs=moralstory foundation={args.foundation!r}")
        pos_prompts, neg_prompts = make_moralstory_pairs(
            tok, n_pairs=args.n_pairs, foundation=args.foundation, thinking=True)
    elif args.pairs_source == "persona_library":
        pos_prompts, neg_prompts, persona_library_meta = make_persona_library_pairs(
            tok,
            library_dir=args.persona_library_dir,
            n_pairs=args.n_pairs,
            pair_id=args.persona_library_pair,
            template=args.persona_library_template,
            scenario_path=args.persona_library_scenarios,
            thinking=True,
        )
        vec_label = (f"{args.method}: {persona_library_meta['pair_id']} "
                     f"({persona_library_meta['pos_persona']} vs {persona_library_meta['neg_persona']})")
    else:
        persona_pairs, template = PERSONA_REGISTRY[args.persona]
        pos_pole, neg_pole = persona_pairs[0]
        vec_label = f"{args.method}: {args.persona} ({pos_pole} vs {neg_pole})"
        logger.info(f"persona={args.persona} template={template!r} +pole={pos_pole!r} -pole={neg_pole!r}")
        pos_prompts, neg_prompts = make_persona_pairs(
            tok, n_pairs=args.n_pairs, thinking=True, persona_pairs=persona_pairs, template=template)
    if args.pairs_source == "persona_library" and args.persona_library_scenarios is not None:
        calib_prompts = _persona_library_calib_prompts(tok, args.persona_library_scenarios)
        logger.info(f"calibration prompts: 8 selected-scenario IID short/mid/long + 4 OOD")
    else:
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
                     "persona": args.persona, "vec_label": vec_label,
                     "layers": list(layers), "calibrated_C": C, "kl_p95_at_calib": kl_hit,
                     "instruments": {}}
    if persona_library_meta is not None:
        summary["persona_library"] = persona_library_meta

    # === ordinal instruments: administer over the signed c-sweep =========
    # poles: (tag, coeff, c_mult). c_mult is the SIGNED multiplier of calibrated C written to the CSV
    # so the plotter draws the real trajectory; base is c=0. A '1' grid gives the classic base/+C/-C.
    mults = [float(m) for m in args.c_grid.split(",")]
    poles = [("base", None, 0.0)]
    for m in mults:
        poles += [(f"pos{m:g}", +m * C, +m), (f"neg{m:g}", -m * C, -m)]
    for name in [n for n in args.instruments if n in ORDINAL_INSTRUMENTS]:
        instr = get_instrument(name)
        logger.info(f"\n=== administer {name} ({instr.display}, {len(instr.dimensions)} factors) "
                    f"over signed c-mults {[cm for _, _, cm in poles]} (calibrated C={C:+.3f}) ===")
        prof_by_pole = {}
        for tag, coeff, _cm in poles:
            sample_path = args.out / f"{name}_{tag}_samples.jsonl"
            if coeff is None:
                prof_by_pole[tag] = _administer_profile(model, tok, instr, batch_size=args.admin_batch_size,
                                                        max_think_tokens=args.admin_think_tokens,
                                                        n_samples=args.admin_n_samples,
                                                        temperature=args.admin_temperature,
                                                        top_p=args.admin_top_p,
                                                        sample_path=sample_path, pole=tag, c=_cm)
            else:
                with v(model, C=coeff):
                    prof_by_pole[tag] = _administer_profile(model, tok, instr, batch_size=args.admin_batch_size,
                                                            max_think_tokens=args.admin_think_tokens,
                                                            n_samples=args.admin_n_samples,
                                                            temperature=args.admin_temperature,
                                                            top_p=args.admin_top_p,
                                                            sample_path=sample_path, pole=tag, c=_cm)
            pm = list(prof_by_pole[tag].values())[0]["pmass"]
            logger.info(f"  {name} {tag} (C={coeff}): pmass={pm:.3f} "
                        f"C=" + ", ".join(f"{f}={prof_by_pole[tag][f]['C']:+.2f}" for f in instr.dimensions))
        # mean = E (human-comparable), C = logit contrast (steer-legible), logodds = direction.
        rows = [{"foundation": f, "pole": tag, "c": cm,
                 "mean": prof_by_pole[tag][f]["E"], "C": prof_by_pole[tag][f]["C"],
                 "E_sd": prof_by_pole[tag][f]["E_sd"],
                 "E_ci95_lo": prof_by_pole[tag][f]["E_ci95_lo"],
                 "E_ci95_hi": prof_by_pole[tag][f]["E_ci95_hi"],
                 "C_sd": prof_by_pole[tag][f]["C_sd"],
                 "C_ci95_lo": prof_by_pole[tag][f]["C_ci95_lo"],
                 "C_ci95_hi": prof_by_pole[tag][f]["C_ci95_hi"],
                 "framing_spread": prof_by_pole[tag][f]["framing_spread"],
                 "logodds": prof_by_pole[tag][f]["logodds"],
                 "pmass": prof_by_pole[tag][f]["pmass"]}
                for tag, _, cm in poles for f in instr.dimensions]
        with open(args.out / f"{name}_profiles.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "foundation", "pole", "c", "mean", "E_sd", "E_ci95_lo", "E_ci95_hi",
                "C", "C_sd", "C_ci95_lo", "C_ci95_hi", "framing_spread", "logodds", "pmass"])
            w.writeheader()
            w.writerows(rows)
        summary["instruments"][name] = {"display": instr.display, "dimensions": instr.dimensions,
                                        "pmass_base": prof_by_pole["base"][instr.dimensions[0]]["pmass"]}

    # === nominal MFV: evaluate over the same signed c-sweep ================
    if "mfv" in args.instruments:
        logger.info("\n=== evaluate MFV (classic vignettes) over signed c-mults ===")
        # verbose=0 + log_demo=False: BOTH bs=1 demo traces must be off. After the run
        # accumulates state (train + ordinal admin), a bs=1 large-budget forced-choice /
        # free-generation NaNs and that NaN forward poisons the subsequent batched MFV
        # eval (Qwen3.5 gated-delta-net recurrent-state persistence) -- the jobs-183/210/213
        # collapse (pmass 0.166). There are TWO such demos: the adapter's _log_eval_demo_trace
        # (log_demo) and tinymfv.evaluate's internal free_generation_demo (verbose>=1). The
        # no-demo path is coherent (pmass 0.984, bisect 206/208/stage3, all verbose=0). The
        # demos are logging niceties, not the measurement, so drop both here.
        mfv_kw = dict(name="classic", log_demo=False, verbose=0,
                      max_think_tokens=args.max_think_tokens, batch_size=args.eval_batch_size)
        # Coherence under forced reads: pmass is pinned high by the prefill scaffold, so
        # read breakage off frac_unscorable (self-close rate, ~0 once tokens are suppressed)
        # and mean_margin (healthy ~1-3 nats, -> 0 when steering destroys the model). Saved
        # per pole so the figure is self-documenting (was dropped before, not recoverable).
        def coh(rep):
            i = rep["info"]
            return {"mean_margin": rep["mean_margin"], "frac_unscorable": i["frac_unscorable"],
                    "mean_pmass_allowed": i["mean_pmass_allowed"], "mean_nll_prefill": i["mean_nll_prefill"]}
        mfv_reports = {}
        for tag, coeff, cm in poles:
            if coeff is None:
                mfv_reports[tag] = evaluate_multibool(model, tok, **mfv_kw)
            else:
                with v(model, C=coeff):
                    mfv_reports[tag] = evaluate_multibool(model, tok, **mfv_kw)
            c = coh(mfv_reports[tag])
            logger.info(f"  MFV {tag} c={cm:+g} (C={coeff}): margin={c['mean_margin']:+.2f}nat "
                        f"unscorable={c['frac_unscorable']:.3f} pmass={c['mean_pmass_allowed']:.3f} "
                        f"nll_prefill={c['mean_nll_prefill']:.2f}")
        base_report = mfv_reports["base"]
        base_logit = baseline_logit_per_foundation(base_report)
        mfv_dlogit = {
            tag: ({f: {"mean": 0.0, "std": 0.0, "sem": 0.0, "n": base_logit[f]["n"],
                       "n_total": base_logit[f]["n_total"]} for f in FOUNDATION_ORDER}
                  if tag == "base" else dlogit_per_foundation(base_report, rep))
            for tag, rep in mfv_reports.items()
        }
        rows = []
        for tag, _coeff, cm in poles:
            cinfo = coh(mfv_reports[tag])
            for f in FOUNDATION_ORDER:
                dl = mfv_dlogit[tag][f]
                rows.append({
                    "foundation": f, "pole": tag, "c": cm,
                    "mean": base_logit[f]["mean"] + dl["mean"],
                    "dlogit": dl["mean"], "dlogit_sd": dl["std"], "dlogit_sem": dl["sem"],
                    "pmass": cinfo["mean_pmass_allowed"],
                    "mean_margin": cinfo["mean_margin"],
                    "frac_unscorable": cinfo["frac_unscorable"],
                    "mean_nll_prefill": cinfo["mean_nll_prefill"],
                })
        with open(args.out / "mfv_profiles.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=[
                "foundation", "pole", "c", "mean", "dlogit", "dlogit_sd", "dlogit_sem",
                "pmass", "mean_margin", "frac_unscorable", "mean_nll_prefill"])
            w.writeheader()
            w.writerows(rows)
        for tag, rep in mfv_reports.items():
            c = coh(rep)
            logger.info(f"  MFV summary {tag}: margin={c['mean_margin']:+.2f}nat pmass={c['mean_pmass_allowed']:.3f}")
        logger.info("  MFV base logit: " + ", ".join(f"{f}={format_cell(base_logit[f])}" for f in FOUNDATION_ORDER))
        assert "pos1" in mfv_reports and "neg1" in mfv_reports, "mfv.json compatibility needs c-grid to include 1"
        (args.out / "mfv.json").write_text(json.dumps({
            "base_logit_per_foundation": base_logit,
            "pos": {"coeff": +C, "dlogit_per_foundation": mfv_dlogit["pos1"]},
            "neg": {"coeff": -C, "dlogit_per_foundation": mfv_dlogit["neg1"]},
            "foundation_order": list(FOUNDATION_ORDER),
            "coherence": {"base": coh(base_report), "pos": coh(mfv_reports["pos1"]), "neg": coh(mfv_reports["neg1"])},
        }, indent=2))
        summary["instruments"]["mfv"] = {"display": "MFV vignettes", "foundations": list(FOUNDATION_ORDER)}

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    append_run(args.out, {**meta, "kind": "allinstr", "method": args.method, "calibrated_C": C})
    logger.info(f"\n=== allinstr showcase complete -> {args.out} ===")
    logger.info("SHOULD: one <instr>_profiles.csv per instrument over the same signed c-grid, "
                "plus mfv.json compatibility dump. ELSE pmass<<1 means steering broke the readout.")


if __name__ == "__main__":
    main()
