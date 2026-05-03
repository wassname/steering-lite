"""vgel/repeng baseline -- raw uncalibrated coefficient.

Runs https://github.com/vgel/repeng on the same persona-branching pairs the
sweep uses, at the same layer set, but WITHOUT iso-KL calibration. The
purpose is to show what unprincipled coefficient choice looks like next to
the calibrated steering-lite methods. Compare side-by-side via Δlogit per
foundation.

Why no calibration: vgel/repeng ships with `coeff=1.0` (sometimes 1.5 or 2.0
in their examples) chosen by hand; that's the de-facto baseline most papers
compare to. Calibrating it would defeat the point.

Pipeline:
    1. build POS/NEG persona-branching pairs (shared with sweep).
    2. train repeng `ControlVector` on the same pair list.
    3. wrap model with `ControlModel`, apply control at args.coeff (default 1.5).
    4. eval on tinymfv airisk vignettes.
    5. paired Δlogit vs cached bare baseline; same JSON schema as sweep methods.

install: `uv sync --extra baseline` (adds repeng to deps).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from loguru import logger
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_lite._quiet import quiet_external_logs
from _meta import make_metadata, append_run
from steering_lite.data import (
    PERSONA_PAIRS_AUTHORITY, PROMPT_TEMPLATE, load_suffixes,
)
from steering_lite.eval.foundations import (
    FOUNDATION_ORDER, FOUNDATION_SHORT,
    axis_shift, cue, dlogit_per_foundation, format_cell,
)
from steering_lite.eval.tinymfv import evaluate_with_vector


def _build_repeng_dataset(tok, n_pairs: int, seed: int = 42):
    """repeng wants a list of `DatasetEntry(positive=str, negative=str)`.

    We re-use the steering-lite persona-branching scheme: same suffix +
    user_msg, only the system persona differs. repeng's own template is
    looser (it uses a single string with no chat template), so we materialise
    the chat-template ourselves and hand repeng the rendered strings.
    """
    from repeng import DatasetEntry
    import random

    rng = random.Random(seed)
    entries = load_suffixes(thinking=True)
    n = min(n_pairs, len(entries))
    sampled = rng.sample(entries, n)

    pos_personas = [p for p, _ in PERSONA_PAIRS_AUTHORITY]
    neg_personas = [n for _, n in PERSONA_PAIRS_AUTHORITY]

    out = []
    for entry in sampled:
        suffix = entry["suffix"]
        user_msg = entry["user_msg"]
        pos_sys = PROMPT_TEMPLATE.format(persona=rng.choice(pos_personas))
        neg_sys = PROMPT_TEMPLATE.format(persona=rng.choice(neg_personas))
        pos = tok.apply_chat_template(
            [{"role": "system", "content": pos_sys},
             {"role": "user", "content": user_msg},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True)
        neg = tok.apply_chat_template(
            [{"role": "system", "content": neg_sys},
             {"role": "user", "content": user_msg},
             {"role": "assistant", "content": suffix}],
            tokenize=False, continue_final_message=True)
        out.append(DatasetEntry(positive=pos, negative=neg))
    logger.info(f"repeng dataset: {len(out)} entries")
    return out


def _resolve_layers(model, layers_arg: str) -> list[int]:
    n = model.config.num_hidden_layers
    if layers_arg == "mid":
        # 20% to 80% of depth, but always exclude {0, 1} and {n-1, n-2}.
        lo = max(2, int(n * 0.2))
        hi = min(n - 2, int(n * 0.8))
        return list(range(lo, hi))
    return [int(x) for x in layers_arg.split(",")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--layers", default="mid")
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--coeff", type=float, default=0.75,
                    help="raw control coefficient (uncalibrated)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", default="bfloat16")
    ap.add_argument("--max-think-tokens", type=int, default=256)
    ap.add_argument("--vignettes", default="airisk")
    ap.add_argument("--bare-json", type=Path, default=Path("outputs/tinymfv_sweep/bare.json"),
                    help="reuse bare baseline if present (skip re-running base)")
    ap.add_argument("--out", type=Path, default=Path("outputs/tinymfv_sweep"))
    args = ap.parse_args()

    quiet_external_logs()
    args.out.mkdir(parents=True, exist_ok=True)
    meta = make_metadata(args)
    logger.info(f"run_id={meta['run_id']} commit={meta['git_commit']} ts={meta['timestamp']}")

    # Import repeng here so missing-dep error message is clear.
    from repeng import ControlVector, ControlModel

    logger.info(f"BLUF: repeng baseline coeff={args.coeff} (uncalibrated) "
                f"on {args.model} with persona-branching pairs.")
    logger.info("EXPECT: nonzero axis_shift but unpredictable magnitude (no KL bound). "
                "If vector is bad, expect saturation -> gibberish -> Δlogit≈0 across all "
                "foundations because pmass collapses.")

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # repeng's ControlModel wrapper trips an `s_aux is None` path inside
    # transformers' flash_attention_2 integration on Qwen3.5; eager is fine
    # here since repeng's training is short and eval reuses bare baseline.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
    ).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    # --- 1. dataset --------------------------------------------------------
    dataset = _build_repeng_dataset(tok, n_pairs=args.n_pairs)

    # --- 2. train control vector ------------------------------------------
    logger.info("=== repeng ControlVector.train ===")
    t0 = time.time()
    cmodel = ControlModel(model, layer_ids=layers)
    control = ControlVector.train(cmodel, tok, dataset)
    logger.info(f"train elapsed={time.time()-t0:.1f}s")

    # --- 3. base report (reuse if present) --------------------------------
    if args.bare_json.exists():
        logger.info(f"reusing bare report from {args.bare_json}")
        bare = json.loads(args.bare_json.read_text())
        base_report = {"raw": bare["raw_p_true"],
                       "raw_pmass": bare["raw_pmass"]}
    else:
        logger.info("no cached bare report -- running bare eval (with control disabled)")
        cmodel.reset()
        t0 = time.time()
        base_report = evaluate_with_vector(
            model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens)
        logger.info(f"bare elapsed={time.time()-t0:.1f}s")

    # --- 4. eval with control applied --------------------------------------
    logger.info(f"=== repeng eval coeff={args.coeff:+.2f} ===")
    cmodel.set_control(control, coeff=args.coeff)
    t0 = time.time()
    steer_report = evaluate_with_vector(
        cmodel, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens)
    elapsed = time.time() - t0
    cmodel.reset()

    # --- 5. Δlogit + axis_shift -------------------------------------------
    dlogit = dlogit_per_foundation(base_report, steer_report, args.vignettes)
    axis = axis_shift(dlogit)

    # --- 6. persist + print row -------------------------------------------
    out_path = args.out / "repeng.json"
    out_path.write_text(json.dumps({
        "label": "repeng",
        "meta": meta,
        "model": args.model,
        "layers": layers,
        "coeff_raw": args.coeff,
        "n_pairs": args.n_pairs,
        "calibrated": False,
        "vignettes": args.vignettes,
        "dlogit_per_foundation": dlogit,
        "axis_shift": axis,
        "raw_p_true": steer_report["raw"],
        "raw_pmass": steer_report["raw_pmass"],
        "max_think_tokens": args.max_think_tokens,
        "elapsed_s": elapsed,
    }, indent=2))
    append_run(args.out, {**meta, "kind": "baseline", "label": "repeng",
                          "coeff_raw": args.coeff, "axis_shift": axis,
                          "elapsed_s": elapsed})

    from tabulate import tabulate
    headers = (["cue", "axis", "row", "coeff"]
               + [f"Δ{FOUNDATION_SHORT[f]}" for f in FOUNDATION_ORDER]
               + ["t"])
    row = [cue(axis), f"{axis:+.2f}", "repeng_raw", f"{args.coeff:+.2f}"]
    row += [format_cell(dlogit[f]) for f in FOUNDATION_ORDER]
    row.append(f"{elapsed:.0f}s")
    logger.info("\n" + tabulate([row], headers=headers, tablefmt="tsv"))
    logger.info(f"out: {out_path}")


if __name__ == "__main__":
    main()
