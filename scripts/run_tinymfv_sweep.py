"""tinymfv sweep: extract -> calibrate -> eval. Per-foundation Δ-wrongness.

Three baseline modalities + 11 calibrated steering methods, all targeting the
same Care-vs-Traditional/Sanctity axis tinymfv airisk vignettes evaluate:
  1. bare        -- no system prompt, no steering vector
  2. prompt_only -- POS persona as system prompt, no vector
  3. steer_*     -- extract POS-vs-NEG vector, iso-KL calibrate, eval

Persona-branching pairs (POS/NEG share suffix, differ only in 1-2 axis words).
Eval: tinymfv guided CoT, 64 think tokens; we surface the per-foundation
`s_other_violate` table from `analyse()` instead of collapsing to mean(p_true).

Composite metric: axis_shift = ΔSanctity - ΔCare (+ve = moved toward binding,
-ve = moved toward care). Matches the persona axis we steered along.

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
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_TRAD_CARE, PROMPT_TEMPLATE
from steering_lite.eval.tinymfv import evaluate_with_vector


# Display order: target axis first (Care, Sanctity), then remaining 5 foundations.
FOUNDATION_ORDER = ["Care", "Sanctity", "Authority", "Loyalty", "Fairness", "Liberty", "Social Norms"]
FOUNDATION_SHORT = {
    "Care": "Care", "Sanctity": "Sanc", "Authority": "Auth", "Loyalty": "Loy",
    "Fairness": "Fair", "Liberty": "Lib", "Social Norms": "SocN",
}


def _per_foundation(report) -> dict[str, float]:
    """report["table"] -> {foundation_coarse: s_other_violate in [-1, +1]}."""
    df = report["table"]
    return {row["foundation"]: float(row["s_other_violate"]) for _, row in df.iterrows()}


def _delta_per_f(steer_per_f: dict, base_per_f: dict) -> dict[str, float]:
    return {f: steer_per_f.get(f, 0.0) - base_per_f.get(f, 0.0) for f in FOUNDATION_ORDER}


def _axis_shift(delta_per_f: dict[str, float]) -> float:
    """+ve = moved toward Sanctity (traditional), -ve = toward Care."""
    return delta_per_f.get("Sanctity", 0.0) - delta_per_f.get("Care", 0.0)


def _row_for(label: str, base_per_f: dict, steer_per_f: dict, *,
             coeff: str = "n/a", kl: str = "n/a", elapsed_s: float = 0.0) -> list:
    delta = _delta_per_f(steer_per_f, base_per_f)
    axis = _axis_shift(delta)
    cue = "🟢" if abs(axis) > 0.10 else ("🟡" if abs(axis) > 0.03 else "🔴")
    cells = [cue, f"{axis:+.3f}", label, coeff, kl]
    cells += [f"{delta[f]:+.3f}" for f in FOUNDATION_ORDER]
    cells.append(f"{elapsed_s:.0f}s")
    return cells


class _SystemInjectTok:
    """Wraps a tokenizer so apply_chat_template injects a system message.

    Used for the prompt-only baseline: persona delivered as a system prompt,
    no steering vector. Mirrors weight-steering's `prompt_baseline` flag
    (ws/eval/airisk.py:392).
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
    "mean_diff", "mean_centred", "pca", "topk_clusters", "cosine_gated",
    "sspace", "spherical", "directional_ablation", "chars", "linear_act",
    "angular_steering",
]


def _make_cfg(method: str, layers: tuple[int, ...]) -> sl.SteeringConfig:
    common = dict(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0)
    table = {
        "mean_diff":             sl.MeanDiffC(**common),
        "mean_centred":          sl.MeanDiffC(**common, subtract_corpus_mean=True),
        "pca":                   sl.PCAC(**common),
        "topk_clusters":         sl.TopKClustersC(**common, k=4),
        "cosine_gated":          sl.CosineGatedC(**common, tau=0.0),
        "sspace":                sl.SSpaceC(**common, r=8),
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
        # central 50%
        lo, hi = n // 4, (3 * n) // 4
        return tuple(range(lo, hi))
    return tuple(int(x) for x in layers_arg.split(","))


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
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--methods", nargs="+", default=METHODS)
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
    ap.add_argument("--max-think-tokens", type=int, default=64)
    ap.add_argument("--vignettes", default="airisk")
    ap.add_argument("--out", type=Path, default=Path("outputs/tinymfv_sweep"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    logger.info(f"BLUF: model={args.model} methods={len(args.methods)} target_kl={args.target_kl} "
                f"vignettes={args.vignettes} max_think={args.max_think_tokens}")
    logger.info("EXPECT: 3 modalities x airisk vignettes. (1) bare baseline, (2) prompt_only "
                "with POS persona as system prompt, (3) 11 calibrated steering methods.")
    logger.info("EXPECT: axis_shift = ΔSanctity - ΔCare; +ve = moved toward traditional/binding, "
                "-ve = toward care. Per-foundation Δ shown for all 7 foundations.")
    logger.info(f"persona axis: POS='{PERSONA_PAIRS_TRAD_CARE[0][0]}' vs "
                f"NEG='{PERSONA_PAIRS_TRAD_CARE[0][1]}' (and 5 paraphrase pairs)")

    dtype = getattr(torch, args.torch_dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(args.device).eval()

    layers = _resolve_layers(model, args.layers)
    logger.info(f"layers={layers} ({len(layers)} of {model.config.num_hidden_layers})")

    pos_prompts, neg_prompts = make_persona_pairs(
        tok, n_pairs=args.n_pairs, thinking=True,
        persona_pairs=PERSONA_PAIRS_TRAD_CARE,
    )
    calib_prompts = _calib_prompts(tok, n=8)

    rows: list[list] = []

    # === (1) bare baseline: no system prompt, no steering vector ===========
    logger.info("\n=== bare baseline (no steering, no system prompt) ===")
    base_t0 = time.time()
    base_report = evaluate_with_vector(model, tok, name=args.vignettes,
                                       max_think_tokens=args.max_think_tokens)
    base_per_f = _per_foundation(base_report)
    logger.info(f"bare per-foundation s_other_violate: " +
                ", ".join(f"{f}={base_per_f.get(f, 0):+.3f}" for f in FOUNDATION_ORDER))
    logger.info(f"bare baseline elapsed={time.time()-base_t0:.1f}s "
                f"wrongness_mean={base_report['wrongness']:+.3f}")
    # bare row: per-foundation Δ vs itself = 0, so we display absolute s_other_violate
    bare_row = ["⚪", "+0.000", "bare", "n/a", "n/a"]
    bare_row += [f"{base_per_f.get(f, 0):+.3f}" for f in FOUNDATION_ORDER]
    bare_row.append(f"{time.time()-base_t0:.0f}s")
    rows.append(bare_row)

    # === (2) prompt_only baseline: POS persona as system prompt ============
    if args.prompt_baseline:
        pos_persona = PERSONA_PAIRS_TRAD_CARE[0][0]
        sys_prompt = PROMPT_TEMPLATE.format(persona=pos_persona)
        logger.info(f"\n=== prompt_only (system='{sys_prompt}') ===")
        wrapped_tok = _SystemInjectTok(tok, sys_prompt)
        pb_t0 = time.time()
        pb_report = evaluate_with_vector(model, wrapped_tok, name=args.vignettes,
                                         max_think_tokens=args.max_think_tokens)
        pb_per_f = _per_foundation(pb_report)
        rows.append(_row_for("prompt_only", base_per_f, pb_per_f,
                             elapsed_s=time.time() - pb_t0))

    # === (3) 11 calibrated steering methods =================================
    for method in tqdm(args.methods, desc="methods", mininterval=60):
        try:
            cfg = _make_cfg(method, layers)
        except KeyError:
            logger.warning(f"skip unknown method {method!r}")
            continue
        logger.info(f"\n=== steer_{method} ===")
        t0 = time.time()
        v = sl.train(model, tok, pos_prompts, neg_prompts, cfg,
                     batch_size=args.batch_size, max_length=args.max_length)
        coeff_calib, _hist = sl.calibrate_iso_kl(
            v, model, tok, calib_prompts,
            target_kl=args.target_kl, T=args.calib_T,
            max_iters=args.calib_iters, device=args.device,
        )
        v.cfg.coeff = float(coeff_calib)
        kl_hit = _hist[-1].get("kl_p95", float("nan")) if _hist else float("nan")

        with v(model):
            steer_report = evaluate_with_vector(
                model, tok, name=args.vignettes, max_think_tokens=args.max_think_tokens, vector=v)
        steer_per_f = _per_foundation(steer_report)
        delta_per_f = _delta_per_f(steer_per_f, base_per_f)
        elapsed = time.time() - t0

        rows.append(_row_for(f"steer_{method}", base_per_f, steer_per_f,
                             coeff=f"{coeff_calib:+.3f}", kl=f"{kl_hit:.2f}",
                             elapsed_s=elapsed))

        out_path = args.out / f"{method}.json"
        out_path.write_text(json.dumps({
            "method": method, "model": args.model, "layers": list(layers),
            "coeff_calibrated": float(coeff_calib), "target_kl": args.target_kl,
            "kl_p95_at_calib": kl_hit,
            "base_per_foundation": base_per_f,
            "steer_per_foundation": steer_per_f,
            "delta_per_foundation": delta_per_f,
            "axis_shift": _axis_shift(delta_per_f),
            "n_pairs": args.n_pairs, "max_think_tokens": args.max_think_tokens,
            "vignettes": args.vignettes, "elapsed_s": elapsed,
        }, indent=2))

    logger.info("\n=== tinymfv sweep complete ===")
    logger.info(f"out: {args.out}")
    logger.info("SHOULD: bare row shows absolute s_other_violate per foundation (Care should "
                "be high, Sanctity low). Other rows show Δ vs bare. axis_shift > 0 means we "
                "moved toward traditional/binding cluster.")
    headers = (["cue", "axis", "row", "C_calib", "kl_p95"]
               + [f"Δ{FOUNDATION_SHORT[f]}" for f in FOUNDATION_ORDER]
               + ["t"])
    logger.info("\n" + tabulate(rows, headers=headers, tablefmt="tsv"))


if __name__ == "__main__":
    main()
