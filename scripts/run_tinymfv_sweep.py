"""tinymfv sweep: extract -> calibrate -> eval. Per-foundation Δlogit ± std.

Three baseline modalities + 11 calibrated steering methods, all targeting the
Authority↓ + Care↑ axis (Forethought "AI character" framing) on tinymfv airisk
vignettes:
  1. bare        -- no system prompt, no steering vector
  2. prompt_only -- POS persona as system prompt, no vector
  3. steer_*     -- extract POS-vs-NEG vector, iso-KL calibrate, eval at ±C

Persona-branching pairs (POS/NEG share suffix, differ only in 1-2 axis words):
POS = ethical-pushback/stakeholder-care, NEG = unquestioning-compliance. Both
foundations move together (questions authority *because* of harm to people)
so the steering direction is interpretable.

Eval: tinymfv guided CoT, 64 think tokens; per-foundation Δlogit (paired by
(vid, cond)) ± std across pairs. Bidirectional: every method's |C| is iso-KL
calibrated once, then we eval at +C and -C and save both reports. The
aggregator picks SI_best per foundation across signs; eval (not a probe)
decides direction.

Composite metric: axis_shift = ΔlogitSanctity - ΔlogitCare nats (legacy
Care/Sanctity composite kept for cross-axis comparison; the new headline is
Auth+Care, computed by the aggregator).

Refs:
  calibration: https://gist.github.com/wassname/6c11cf30b43d8c228bc114795f1019c7
  guided gen:  https://gist.github.com/wassname/733c568cd29c2a402be4442d6a061899
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
from _meta import make_metadata, append_run
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY, PROMPT_TEMPLATE
from steering_lite.eval.tinymfv import evaluate_with_vector
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, FOUNDATION_SHORT,
    baseline_logit_per_foundation, dlogit_per_foundation,
    flips_per_foundation, axis_shift, format_cell, cue,
)


def _row_steer(label: str, dlogit_per_f: dict, *,
               coeff: str = "n/a", kl: str = "n/a", elapsed_s: float = 0.0) -> list:
    """Row for prompt_only / steer_* / repeng / engineered_prompt: paired Δlogit cells."""
    axis = axis_shift(dlogit_per_f)
    cells = [cue(axis), f"{axis:+.2f}", label, coeff, kl]
    cells += [format_cell(dlogit_per_f[f]) for f in FOUNDATION_ORDER]
    cells.append(f"{elapsed_s:.0f}s")
    return cells


def _row_bare(absolute_logit_per_f: dict, *, elapsed_s: float = 0.0) -> list:
    """Bare row shows absolute logit(wrongness) per foundation -- the reference,
    not a Δ. High Care + low Sanctity is the expected starting point."""
    cells = ["⚪", "ref", "bare", "n/a", "n/a"]
    cells += [format_cell(absolute_logit_per_f[f]) for f in FOUNDATION_ORDER]
    cells.append(f"{elapsed_s:.0f}s")
    return cells


class _SystemInjectTok:
    """Wraps a tokenizer so apply_chat_template injects a system message.

    Used for prompt-only / engineered-prompt baselines: persona delivered as a
    system prompt, no steering vector.
    """
    def __init__(self, tok, system: str):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_sys", system)

    def __getattr__(self, name):
        return getattr(self._tok, name)

    def __setattr__(self, name, value):
        setattr(self._tok, name, value)

    def __call__(self, *args, **kw):
        return self._tok(*args, **kw)

    def apply_chat_template(self, messages, **kw):
        if messages and messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self._sys}] + list(messages)
        return self._tok.apply_chat_template(messages, **kw)

quiet_external_logs()
logger.remove()
logger.add(lambda x: tqdm.write(x, end=""), level="INFO", colorize=False, format="{message}")


METHODS = [
    "sspace", "sspace_ablate", "sspace_damp_amp", "super_sspace",
    "mean_diff", "mean_centred", "pca", "topk_clusters", "cosine_gated",
    "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering",
]

def _make_cfg(method: str, layers: tuple[int, ...], *,
              sspace_r: int = -1, sspace_target_submodule: str | None = None) -> sl.SteeringConfig:
    common = dict(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0)
    sspace_kw: dict = {"r": sspace_r}
    if sspace_target_submodule is not None:
        sspace_kw["target_submodule"] = sspace_target_submodule
    table = {
        "mean_diff":             sl.MeanDiffC(**common),
        "mean_centred":          sl.MeanDiffC(**common, subtract_corpus_mean=True),
        "pca":                   sl.PCAC(**common),
        "topk_clusters":         sl.TopKClustersC(**common, k=4),
        "cosine_gated":          sl.CosineGatedC(**common, tau=0.0),
        "sspace":                sl.SSpaceC(**common, **sspace_kw),
        "sspace_ablate":         sl.SSpaceAblateC(**common, **sspace_kw),
        "sspace_damp_amp":       sl.SSpaceDampAmpC(**common, **sspace_kw),
        "super_sspace":          sl.SuperSSpaceC(**common, r=sspace_r),
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
        # 20% to 80% of depth, but always exclude {0, 1} and {n-1, n-2}:
        # earliest layers are tokenizer-y, last two are output-projection-bound.
        lo = max(2, int(n * 0.2))
        hi = min(n - 2, int(n * 0.8))
        return tuple(range(lo, hi))
    return tuple(int(x) for x in layers_arg.split(","))


@torch.no_grad()
def _log_persona_demo(model, tok, persona_pairs, user_msgs, *, max_new_tokens: int) -> None:
    """Sanity-check the persona pairs before extract: free-form greedy gen
    from POS and NEG system prompts on shared user_msgs. Shows the full
    chat-templated input + completion *with* special tokens, so we can
    eyeball whether the chat template is right and whether the persona is
    biting (POS vs NEG should differ in voice/values).
    """
    for i, (pos_persona, neg_persona) in enumerate(persona_pairs):
        for j, user_msg in enumerate(user_msgs):
            pos_sys = PROMPT_TEMPLATE.format(persona=pos_persona)
            neg_sys = PROMPT_TEMPLATE.format(persona=neg_persona)
            blocks = []
            for tag, sys_prompt in (("POS", pos_sys), ("NEG", neg_sys)):
                input_ids = tok.apply_chat_template(
                    [{"role": "system", "content": sys_prompt},
                     {"role": "user", "content": user_msg}],
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=False,
                ).to(model.device)
                out = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                )
                full = tok.decode(out[0], skip_special_tokens=False)
                blocks.append(f"--- {tag} (pair={i}, msg={j}) ---\n{full}")
            logger.info(
                "EXPECT: POS+NEG share user_msg + chat template; differ in system persona "
                "and downstream voice. ELSE persona too weak or template misrendered.\n"
                f"=== PERSONA demo (pair={i}/{len(persona_pairs)}, user_msg={user_msg!r}) ===\n"
                + "\n".join(blocks)
                + "\n=== /PERSONA ==="
            )


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
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--methods", nargs="*", default=METHODS,
                    help="If empty list passed, runs only bare + prompt_only baselines.")
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--n-pairs", type=int, default=256,
                    help="contrastive pairs from data/branching_suffixes.json (~550 max)")
    ap.add_argument("--prompt-baseline", action=argparse.BooleanOptionalAction, default=True,
                    help="include a persona-as-system-prompt baseline row (no steering vector)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--target-kl", type=float, default=1.0)
    ap.add_argument("--calib-T", type=int, default=20)
    ap.add_argument("--calib-iters", type=int, default=8)
    ap.add_argument("--max-think-tokens", type=int, default=256)
    ap.add_argument("--vignettes", default="airisk")
    ap.add_argument("--sspace-r", type=int, default=-1,
                    help="rank for sspace* variants. -1 = full rank.")
    ap.add_argument("--sspace-target-submodule", default=None,
                    help="regex matched against block.named_modules() for sspace* variants. "
                         "None uses the variant default (residual writers: mlp.down_proj + self_attn.o_proj).")
    ap.add_argument("--out", type=Path, default=Path("outputs/tinymfv_sweep"))
    ap.add_argument("--demo-only", action="store_true",
                    help="run persona demo, then exit before bare/sweep")
    ap.add_argument("--demo-pairs", type=int, default=2,
                    help="number of user_msgs to demo each persona pair on")
    ap.add_argument("--demo-max-tokens", type=int, default=200)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(args)
    logger.info(f"run_id={meta['run_id']} commit={meta['git_commit']} ts={meta['timestamp']}")

    logger.info(f"BLUF: model={args.model} methods={len(args.methods)} target_kl={args.target_kl} "
                f"vignettes={args.vignettes} max_think={args.max_think_tokens}")
    logger.info(f"EXPECT: 3 modalities x {args.vignettes} vignettes. (1) bare baseline, (2) prompt_only "
                "with POS persona as system prompt, (3) 11 calibrated steering methods.")
    logger.info("EXPECT: axis_shift = ΔlogitSanctity - ΔlogitCare nats (legacy composite). "
                "Headline axis is Auth↓ -- aggregate_flips.py reports SI on Authority.")
    logger.info(f"persona axis: POS='{PERSONA_PAIRS_AUTHORITY[0][0]}' vs "
                f"NEG='{PERSONA_PAIRS_AUTHORITY[0][1]}' "
                f"(+{len(PERSONA_PAIRS_AUTHORITY)-1} paraphrase pairs)")

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # No flash_attention_2: Qwen3.5 + FA2 trips an `s_aux is None` path in
    # transformers' integration. sdpa (default) works; speed cost is small
    # since eval is short-suffix-bound, not prefill-bound.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    pos_prompts, neg_prompts = make_persona_pairs(
        tok, n_pairs=args.n_pairs, thinking=True,
        persona_pairs=PERSONA_PAIRS_AUTHORITY,
    )
    calib_prompts = _calib_prompts(tok, n=8)

    # Persona sanity check: POS/NEG free-form gen on shared user_msgs, with
    # full chat-templated input + special tokens visible. Run *before* extract
    # so we can abort early if personas don't bite.
    _log_persona_demo(
        model, tok, PERSONA_PAIRS_AUTHORITY,
        user_msgs=_calib_prompts(tok, n=args.demo_pairs, seed=1),
        max_new_tokens=args.demo_max_tokens,
    )
    if args.demo_only:
        logger.info("--demo-only set; exiting before extract.")
        return

    # NB: "Think briefly. " is prepended to the JSON instruction in
    # tinymfv.FRAMES["q"] (see steering_lite.eval.tinymfv). One place,
    # applied uniformly to every eval below -- no system-prompt injection.

    rows: list[list] = []
    method_summaries: list[dict] = []

    # === (1) bare baseline: no system prompt, no steering ==================
    logger.info("\n=== bare baseline (no system prompt, no steering) ===")
    base_t0 = time.time()
    base_report = evaluate_with_vector(model, tok, name=args.vignettes,
                                       max_think_tokens=args.max_think_tokens)
    base_logit_per_f = baseline_logit_per_foundation(base_report, args.vignettes)
    bare_elapsed = time.time() - base_t0
    logger.info("bare per-foundation logit(wrongness) ± std: " +
                ", ".join(f"{f}={format_cell(base_logit_per_f[f])}" for f in FOUNDATION_ORDER))
    logger.info(f"bare elapsed={bare_elapsed:.1f}s "
                f"wrongness_mean={base_report['wrongness']:+.3f}")
    rows.append(_row_bare(base_logit_per_f, elapsed_s=bare_elapsed))

    # Persist bare report for later re-analysis (raw p_true grid + bool_mass).
    # raw_pmass MUST be saved -- aggregator's pmass-floor gating uses it on the
    # bare side; without it, gating is asymmetric (bare cells default to 1.0,
    # steered cells get gated, biasing |Δlogit| of heavily-steered methods).
    (args.out / "bare.json").write_text(json.dumps({
        "label": "bare",
        "meta": meta,
        "model": args.model, "vignettes": args.vignettes,
        "max_think_tokens": args.max_think_tokens,
        "wrongness_mean": float(base_report["wrongness"]),
        "absolute_logit_per_foundation": base_logit_per_f,
        "raw_p_true": base_report["raw"],
        "raw_pmass": base_report["raw_pmass"],
        "info": {k: v for k, v in base_report["info"].items() if k != "elapsed_s"},
        "elapsed_s": bare_elapsed,
    }, indent=2))
    method_summaries.append({"label": "bare", "elapsed_s": bare_elapsed,
                             "wrongness_mean": float(base_report["wrongness"])})

    # === (2) prompt_only baseline: POS persona as system prompt ============
    if args.prompt_baseline:
        pos_persona = PERSONA_PAIRS_AUTHORITY[0][0]
        sys_prompt = PROMPT_TEMPLATE.format(persona=pos_persona)
        logger.info(f"\n=== prompt_only (system='{sys_prompt}') ===")
        wrapped_tok = _SystemInjectTok(tok, sys_prompt)
        pb_t0 = time.time()
        pb_report = evaluate_with_vector(model, wrapped_tok, name=args.vignettes,
                                         max_think_tokens=args.max_think_tokens)
        pb_dlogit = dlogit_per_foundation(base_report, pb_report, args.vignettes)
        pb_flips = flips_per_foundation(base_report, pb_report, args.vignettes)
        pb_elapsed = time.time() - pb_t0
        rows.append(_row_steer("prompt_only", pb_dlogit, elapsed_s=pb_elapsed))

        (args.out / "prompt_only.json").write_text(json.dumps({
            "label": "prompt_only",
            "meta": meta,
            "model": args.model,
            "system_prompt": sys_prompt, "vignettes": args.vignettes,
            "dlogit_per_foundation": pb_dlogit,
            "flips_per_foundation": pb_flips,
            "axis_shift": axis_shift(pb_dlogit),
            "raw_p_true": pb_report["raw"],
            "raw_pmass": pb_report.get("raw_pmass", {}),
            "elapsed_s": pb_elapsed,
        }, indent=2))
        method_summaries.append({"label": "prompt_only",
                                 "axis_shift": axis_shift(pb_dlogit),
                                 "elapsed_s": pb_elapsed})

    # === (3) 11 calibrated steering methods =================================
    # Bidirectional: calibrate |C| at +sign, then run eval at +C and -C. Lets the
    # eval (not a sign-probe) decide which direction is the intended one. KL at
    # -C is not exactly the same as at +C, but at target_kl=1.0 the asymmetry is
    # small; revisit with dual calibration if it bites.
    for method in tqdm(args.methods, desc="methods", mininterval=60):
        cfg = _make_cfg(method, layers,
                        sspace_r=args.sspace_r,
                        sspace_target_submodule=args.sspace_target_submodule)
        logger.info(f"\n=== steer_{method} ===")
        t0 = time.time()
        v = sl.train(model, tok, pos_prompts, neg_prompts, cfg,
                     batch_size=args.batch_size, max_length=args.max_length)
        coeff_calib, _hist = sl.calibrate_iso_kl(
            v, model, tok, calib_prompts,
            target_kl=args.target_kl, T=args.calib_T,
            max_iters=args.calib_iters, device=args.device,
        )
        kl_hit = _hist[-1].get("kl_p95", float("nan")) if _hist else float("nan")
        C = float(coeff_calib)

        # +C eval
        v.cfg.coeff = +C
        with v(model):
            pos_report = evaluate_with_vector(
                model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens, vector=v)
        pos_dlogit = dlogit_per_foundation(base_report, pos_report, args.vignettes)
        pos_flips = flips_per_foundation(base_report, pos_report, args.vignettes)

        # -C eval (same |C|, flipped sign — no recalibration)
        v.cfg.coeff = -C
        with v(model):
            neg_report = evaluate_with_vector(
                model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens, vector=v)
        neg_dlogit = dlogit_per_foundation(base_report, neg_report, args.vignettes)
        neg_flips = flips_per_foundation(base_report, neg_report, args.vignettes)

        elapsed = time.time() - t0
        # Pick the sign with larger |axis_shift| for the headline row; the JSON
        # keeps both so the aggregator can compute SI_best per foundation.
        ax_pos, ax_neg = axis_shift(pos_dlogit), axis_shift(neg_dlogit)
        if abs(ax_pos) >= abs(ax_neg):
            best_label, best_dlogit, best_C = f"steer_{method}[+]", pos_dlogit, +C
        else:
            best_label, best_dlogit, best_C = f"steer_{method}[-]", neg_dlogit, -C
        rows.append(_row_steer(best_label, best_dlogit,
                               coeff=f"{best_C:+.3f}", kl=f"{kl_hit:.2f}",
                               elapsed_s=elapsed))

        out_path = args.out / f"{method}.json"
        out_path.write_text(json.dumps({
            "label": f"steer_{method}",
            "meta": meta,
            "method": method, "model": args.model, "layers": list(layers),
            "calibrated_C": C, "target_kl": args.target_kl,
            "kl_p95_at_calib": kl_hit,
            "pos": {
                "coeff": +C,
                "dlogit_per_foundation": pos_dlogit,
                "flips_per_foundation": pos_flips,
                "axis_shift": ax_pos,
                "raw_p_true": pos_report["raw"],
                "raw_pmass": pos_report["raw_pmass"],
            },
            "neg": {
                "coeff": -C,
                "dlogit_per_foundation": neg_dlogit,
                "flips_per_foundation": neg_flips,
                "axis_shift": ax_neg,
                "raw_p_true": neg_report["raw"],
                "raw_pmass": neg_report.get("raw_pmass", {}),
            },
            "n_pairs": args.n_pairs, "max_think_tokens": args.max_think_tokens,
            "vignettes": args.vignettes, "elapsed_s": elapsed,
        }, indent=2))
        method_summaries.append({
            "label": f"steer_{method}",
            "calibrated_C": C, "kl_p95_at_calib": kl_hit,
            "axis_shift_pos": ax_pos, "axis_shift_neg": ax_neg,
            "elapsed_s": elapsed,
        })

    # One audit line per sweep invocation. Metadata is per-run, not per-method;
    # method JSONs already carry their full results, so this is a thin index.
    append_run(args.out, {**meta, "kind": "sweep", "methods": method_summaries})

    logger.info("\n=== tinymfv sweep complete ===")
    logger.info(f"out: {args.out}")
    logger.info(f"runs.jsonl: {args.out / 'runs.jsonl'} (run_id={meta['run_id']})")
    logger.info("SHOULD: bare row shows absolute logit(wrongness)±std per foundation; expect "
                "Care high (model thinks care violations are wrong), Sanctity lower. Other rows "
                "are paired Δlogit±std vs bare. axis_shift>0 means moved toward binding cluster.")
    headers = (["cue", "axis", "row", "C_calib", "kl_p95"]
               + [f"Δ{FOUNDATION_SHORT[f]}" for f in FOUNDATION_ORDER]
               + ["t"])
    logger.info("\n" + tabulate(rows, headers=headers, tablefmt="tsv"))


if __name__ == "__main__":
    main()
