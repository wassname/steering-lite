"""Iso-KL calibration: find the coeff so KL(steer || base) hits a target.

Two functions:
- `measure_kl(...)`: roll out T tokens with steering attached, then teacher-force
  re-score base + steer over the rolled-out sequence, return per-token KL stats.
- `calibrate_iso_kl(...)`: log-log secant solver. Brackets exponentially, then
  interpolates in `(log C, log stat)` space. KL(p_C || p_0) ~ C^2 in the small-C
  regime, so log-log slope is ~2 and secant converges in ~3-5 iters (vs ~10 for
  pure bisection).

Recommended: seeded sampling with the same random stream at every bracket point,
target_stat="kl_rms". This exposes sampled rollout basins without making the solver
chase unrelated decode noise. (Claude, 2026-07-19)
kl_rms = sqrt(mean per-token KL^2) in nats: whole-distribution, quadratically
tail-weighted (the square inside penalizes tail tokens that derail reasoning),
but reported in nats so it sits directly alongside kl_mean/p95/max rather than
the nats^2 of a raw mean-square.
"""
from __future__ import annotations
import json
import math
import re
from pathlib import Path
from typing import Callable

import torch
from loguru import logger
from tabulate import tabulate
from torch import Tensor
from torch import nn
from tqdm.auto import tqdm

from .config import SteeringConfig
from .vector import Vector


def _log_kl_history(method: str, history: list[dict]) -> None:
    """Tabulate the iso-KL bracket trace once at end of calibrate. Sorted by c
    so monotonicity is visually obvious; 'i' column preserves eval order."""
    if not history:
        return
    indexed = [(i, h) for i, h in enumerate(history)]
    indexed.sort(key=lambda ih: ih[1]["coeff"])
    # include rms: it is the default calibration target (kl_rms), so the table shows the
    # stat we actually solve for, not only mean/p90/p95/max. The steer tail (prompt-0
    # rollout) rides along so degradation is visible in the always-on table, not just
    # the verbose demo. (Claude 2026-07-16)
    rows = [
        [str(i), f"{h['coeff']:+.4f}", f"{h['kl_mean']:.4f}", f"{h['kl_rms']:.4f}",
         f"{h['kl_r4ms4e']:.4f}", f"{h['kl_p95']:.5f}", f"{h['kl_max']:.4f}",
         f"{h.get('rep', 0.0):.2f}", f"{h.get('gen_len', 0.0):.0f}", str(h['n_pos']),
         h["steer_tail"]]
        for i, h in indexed
    ]
    table = tabulate(rows, headers=["i", "c", "mean", "rms", "r4ms4e", "p95", "max",
                                    "rep", "len", "n", "steer tail (prompt 0)"], tablefmt="plain")
    logger.info(
        f"SHOULD: choose the highest C with a coherent tail; a repetition tail "
        f"('but but but') or gibberish marks where the dose is too hot -- read that "
        f"row's r4ms4e as the target_kl for future runs.\n"
        f"--- iso-KL bracket trace ({method}, {len(history)} iters) ---\n{table}")


def _log_per_t_profile(method: str, c: float, per_t_p50: list[float],
                       per_t_p90: list[float], per_t_p95: list[float],
                       per_t_max: list[float], per_t_n: list[int]) -> None:
    """Per-position KL trace at the calibrated c. Lets us see whether KL is
    front-loaded (decreasing/flat across t = T=20 calibration generalizes) or
    ramping (= calibration undershoots inference T)."""
    if not per_t_p95:
        return
    rows = [
        [str(t), f"{p50:.4f}", f"{p90:.4f}", f"{p95:.5f}", f"{mx:.4f}", str(n)]
        for t, (p50, p90, p95, mx, n) in enumerate(
            zip(per_t_p50, per_t_p90, per_t_p95, per_t_max, per_t_n)
        )
    ]
    table = tabulate(rows, headers=["t", "p50", "p90", "p95", "max", "n"], tablefmt="plain")
    logger.info(
        f"SHOULD: per_t_p95 decreasing or flat across t (front-loaded, T calibration "
        f"generalizes). ELSE: ramping -> T undershoots inference.\n"
        f"--- per-t KL profile ({method}, c={c:+.4f}) ---\n{table}"
    )


# Generic prompts for cheap default calibration. Diversity > length: p95 means
# one bad prompt already dominates, so a small diverse set beats many similar
# ones. Coverage spans factual / narrative / technical / multilingual / OOD,
# AND varies the *framing* itself (chat vs chat+assistant_prefix vs raw text)
# because the steering hook fires on every position -- a vector that only
# misbehaves under a weird prefill or on raw (no-template) text is a real
# coherence risk we want measured.
#
# Note on sampling: do_sample is a measure_kl-level knob, not a prompt-level
# one. Mixing sampled and greedy "variants" of the same prompt would just add
# rollout noise to a p95 that is already sensitive to a single bad tail; keep
# rollout determinism uniform and let the prompts carry the diversity.
DEFAULT_MESSAGES_RAW = [
    # chat: math / think
    {"kind": "chat", "user": "Think step by step to calculate the integral of x^2 from 0 to 1 in lean4. ```"},
    # chat: OOD repetition through chat template
    {"kind": "chat", "user": "           fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck fuck"},
    # raw text, no chat template -- tests steering on non-chat token sequences
    {"kind": "raw",  "text": "The 🗼 tower is in Tianducheng, 🇨🇳 天都城 · ℍ𝕒𝕟𝕘𝕫𝕙𝕠𝕦 · ℂ𝕙𝕚𝕟𝕒. `where 🏛️`"},
    # chat: narrative
    {"kind": "chat", "user": "Tell me a greentext story about a small village during the smaller Martion carrot bubble."},
    # raw: non-English, no template
    {"kind": "raw",  "text": "埃菲尔铁塔🗼位于天都城"},
    # chat: multilingual (Esperanto)
    {"kind": "chat", "user": "Kiu estas la ĉefurbo de Francio? Respondu en Antaŭ-Esperanto."},
    # chat: programming
    {"kind": "chat", "user": "Write a Python function that returns the nth Fibonacci number using memoization. `def f(x):\n    return {'': [None, None, None]}```"},
    # chat with assistant prefill -- model sees an unusual start of its own turn
    {"kind": "chat", "user": "Tell me a joke.",
     "assistant_prefix": "<thunk/>Et Tu user</thunk> Here's none, tool(translate\\): 我 我 我 我 我 我 我 我"},
]


def _default_prompts(tok) -> list[Tensor]:
    """Build the default calibration prompt set as pre-tokenized Tensors."""
    out: list[Tensor] = []
    for spec in DEFAULT_MESSAGES_RAW:
        if spec["kind"] == "raw":
            out.append(tok(spec["text"], return_tensors="pt").input_ids[0])
            continue
        msgs = [{"role": "user", "content": spec["user"]}]
        prefix = spec.get("assistant_prefix")
        if prefix:
            msgs.append({"role": "assistant", "content": prefix})
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=False,
                continue_final_message=True, return_tensors="pt",
            ).input_ids[0]
        else:
            ids = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt",
            ).input_ids[0]
        out.append(ids)
    return out


def _tokenize(prompts: list[str] | list[Tensor] | None, tok) -> list[Tensor]:
    if prompts is None:
        return _default_prompts(tok)
    if isinstance(prompts[0], str):
        # apply_chat_template(return_tensors="pt") returns a BatchEncoding
        # in transformers>=4.45; .input_ids[0] gives the (seq_len,) tensor row.
        return [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, return_tensors="pt",
            ).input_ids[0]
            for p in prompts
        ]
    return prompts


@torch.no_grad()
def _kl_per_pos(logp_steer: Tensor, logp_base: Tensor) -> Tensor:
    p_s = logp_steer.exp()
    return (p_s * (logp_steer - logp_base)).sum(dim=-1)


def _ngram_rep(ids: list[int], n: int = 3) -> float:
    """Fraction of repeated n-grams in a token id list: 1 - unique/total. ~1.0 for a
    degenerate 'but but but' loop, ~0 for varied text. Mirrors what NoRepeatNGramLogits-
    Processor acts on, so a -C dose that collapses into repetition is visible pre-eval."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


@torch.no_grad()
def _generate(model, prompt_ids, T, tok, do_sample, device):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = prompt_ids.unsqueeze(0).to(device)
    out = model.generate(
        ids, max_new_tokens=T, pad_token_id=pad_id, eos_token_id=tok.eos_token_id,
        num_return_sequences=1, do_sample=do_sample,
    )
    return out[0, prompt_ids.shape[0]:]


@torch.no_grad()
def measure_kl(
    v: Vector,
    model: nn.Module,
    tok,
    prompts: list[str] | list[Tensor] | None = None,
    *,
    T: int = 20,
    do_sample: bool = False,
    device: str | torch.device = "cuda",
    show_pbar: bool = True,
    log_demo: bool = False,
    demo_log_path: Path | None = None,
    demo_iter: int | None = None,
    verbose_demo: bool = False,
    seed: int | None = None,
) -> dict:
    """Roll out T tokens with steering attached, then score under base
    (detached) and steer (re-attached). Returns KL summary stats and per-token
    KL distributions across positions.

    Logging knobs (method/eval-agnostic — only inputs and outputs):
      log_demo       : emit a BASE-vs-STEER comparison for prompt 0 to loguru.
      demo_log_path  : if set, append one JSONL line per (iter, prompt) with
                       full base text, steer text, and per-position KL.
      demo_iter      : iteration index, written into JSONL records (cosmetic).
      seed           : pair each prompt's sampled RNG stream across calls. (Claude)
    """
    prompts = _tokenize(prompts, tok)
    all_kls = []
    per_t = [[] for _ in range(T)]
    gen_lens, reps = [], []  # per-prompt rollout length + worst 3-gram repetition fraction
    steer_tail = ""  # prompt-0 steered rollout tail, for the always-on bracket table (Claude)
    # The base rollout is display-only: the KL is computed from logp_base vs logp_steer over the
    # SAME steered token ids, so this generation never reaches a statistic. It was produced for
    # every prompt while only prompt 0 is ever printed, so with the usual log_demo=True and no
    # JSONL path, 7 of 8 full T-token rollouts per probe were generated and discarded -- and a
    # probe is the unit calibration spends all its time in. Predicate moved to the use site.
    need_base_gen = demo_log_path is not None or log_demo

    for idx, pids in enumerate(tqdm(prompts, desc="measure_kl",
                                    mininterval=60, disable=not show_pbar)):
        if seed is not None:
            torch.manual_seed(seed + idx)
        with v(model):
            gen = _generate(model, pids, T, tok, do_sample, device)
        n_gen = gen.shape[0]
        if n_gen == 0:
            continue
        gen_lens.append(n_gen)
        reps.append(_ngram_rep(gen.tolist()))
        full_ids = torch.cat([pids.to(device), gen])
        full = full_ids.unsqueeze(0)
        n_p = pids.shape[0]

        logp_base = torch.log_softmax(model(full).logits.float(), dim=-1)[0]
        with v(model):
            logp_steer = torch.log_softmax(model(full).logits.float(), dim=-1)[0]

        slc = slice(n_p - 1, n_p - 1 + n_gen)
        kls = _kl_per_pos(logp_steer[slc], logp_base[slc]).cpu()
        all_kls.append(kls)
        for i in range(n_gen):
            per_t[i].append(float(kls[i]))
        if idx == 0:  # cheap: gen already rolled out for the KL calc (Claude)
            # last ~160 chars of the rollout: long enough to show a repetition loop
            # ("but but but") that only emerges deep into a long-T rollout.
            steer_tail = " ".join(tok.decode(gen, skip_special_tokens=True).split())[-160:]

        # Demo: extra base-only gen for side-by-side text. One per measure_kl
        # call (idx==0) for stdout, all prompts for JSONL. Generate it only when it is actually
        # consumed -- stdout reads prompt 0 only, the JSONL reads all of them.
        if need_base_gen and (demo_log_path is not None or idx == 0):
            base_gen = _generate(model, pids, T, tok, do_sample, device)
            base_full = torch.cat([pids.to(device), base_gen])
            decoded_base = tok.decode(base_full, skip_special_tokens=False)
            decoded_steer = tok.decode(full_ids, skip_special_tokens=False)
            is_final = demo_iter is not None and demo_iter < 0
            # Full BASE vs STEER dump at the FINAL operating point (always) and at every
            # bracket only under verbose_demo. Per-bracket steer tails already ride in the
            # iso-KL table (steer_tail column), so intermediate probes need no console dump.
            if log_demo and idx == 0 and (is_final or verbose_demo):
                stage = ("FINAL operating point" if is_final
                         else f"probe iter {demo_iter} (bracket point, NOT final)")
                logger.info(
                    f"\n=== {v.cfg.method} -> calibrate iso-KL -> {stage} "
                    f"| c={v.cfg.coeff:+.4f} kl_rms[this prompt only]="
                    f"{float(kls.pow(2).mean().sqrt()):.3f} | "
                    f"T={T} tok/KL-probe (<=12 iters) ===\n"
                    "WHAT: same held-out calib prompt at c=0 vs current c. KL calibration "
                    "sweep, NOT the moral eval; c is a bracket point unless FINAL.\n"
                    "The kl_rms above is this one prompt; the value the solver matched to "
                    "target_kl is the whole-prompt-set rms in the bracket table below, and "
                    "the two differ by up to 2x (job 160: 0.349 here vs 0.709 solved).\n"
                    f"--- BASE (c=0) ---\n{decoded_base}\n"
                    f"\n--- STEER ({v.cfg.method}, c={v.cfg.coeff:+.4f}) ---\n{decoded_steer}\n"
                    f"=== /{v.cfg.method} calibrate ==="
                )
            if demo_log_path is not None:
                with demo_log_path.open("a") as f:
                    f.write(json.dumps({
                        "iter": demo_iter,
                        "c": float(v.cfg.coeff),
                        "prompt_idx": idx,
                        "T": T,
                        "base_text": decoded_base,
                        "steer_text": decoded_steer,
                        "per_t_kl": kls.tolist(),
                    }) + "\n")

    cat = torch.cat(all_kls)
    # per-t quantiles: torch.quantile needs >=1 sample per t; pad with 0 if empty.
    def _q(xs, q):
        return float(torch.tensor(xs).quantile(q)) if xs else 0.0
    return {
        "kl_mean": float(cat.mean()),
        # kl_rms = sqrt(mean(KL^2)) in nats: default calibration target. Whole-distribution
        # and quadratically tail-weighted (the square penalizes tail tokens that derail
        # reasoning), but far less noisy than a p95 quantile over few tokens. Reported in
        # nats so target_kl and the table columns are all one unit. (wassname + Claude)
        "kl_rms": float(cat.pow(2).mean().sqrt()),
        # kl_r4ms4e = (mean(KL^4))^(1/4) in nats: like kl_rms but quartically tail-weighted,
        # so a localized front/tail spike that rms dilutes across all tokens still shows. The
        # 4th root keeps the nats scale (no shrink near 0), and it equals max(mean, rms, l4) by
        # the power-mean inequality. New default target: catches the derail spike rms averages
        # away, without a noisy p95 quantile. (wassname + Claude 2026-07-18)
        "kl_r4ms4e": float(cat.pow(4).mean().pow(0.25)),
        "steer_tail": steer_tail,
        "kl_p50": float(cat.quantile(0.50)),
        "kl_p90": float(cat.quantile(0.90)),
        "kl_p95": float(cat.quantile(0.95)),
        "kl_max": float(cat.max()),
        "n_pos": int(cat.numel()),
        "gen_len": (sum(gen_lens) / len(gen_lens)) if gen_lens else 0.0,  # mean rollout length
        "rep": max(reps) if reps else 0.0,   # worst-prompt 3-gram repetition (loop tail)
        "per_t_mean": [sum(xs) / len(xs) if xs else 0.0 for xs in per_t],
        "per_t_p50":  [_q(xs, 0.50) for xs in per_t],
        "per_t_p90":  [_q(xs, 0.90) for xs in per_t],
        "per_t_p95":  [_q(xs, 0.95) for xs in per_t],
        "per_t_max":  [max(xs) if xs else 0.0 for xs in per_t],
        "per_t_n":    [len(xs) for xs in per_t],
    }


def calibrate_iso_kl(
    v: Vector,
    model: nn.Module,
    tok,
    prompts: list[str] | list[Tensor] | None = None,
    *,
    target_kl: float = 0.8,  # highest-coherent-tail kl_rms dose (pca bracket, job 117) (Claude)
    target_stat: str = "kl_rms",
    bracket: tuple[float, float] = (0.001, 256.0),
    tol: float = 0.05,
    max_iters: int = 12,
    T: int = 50,
    device: str | torch.device = "cuda",
    sign: float = 1.0,
    sign_probe: Callable[[Vector], float] | None = None,
    sign_probe_c: float = 1.0,
    demo_log_path: Path | None = None,
    verbose_demo: bool = False,
    seed: int = 0,
    do_sample: bool = True,
) -> tuple[float, list[dict]]:
    """Find coeff C such that stat(C) ~= target_kl using log-log Illinois
    (regula falsi with stale-endpoint reweighting) within a guarded bracket.

    Geometry: KL(p_C || p_0) ≈ ½ C² v^T F v in the small-C regime, so log-log
    slope is ~2 near zero. KL saturates at large C (entropy bound), so the
    log-log curve is concave overall. A plain secant chord on a concave curve
    lies below the curve -> first interpolation overshoots into the high-KL
    side. Illinois halves the stale endpoint's value-weight when one side
    sticks for 2+ iters, giving superlinear convergence without overshoot
    stalling. Bracketing endpoints are always preserved; bisection is the
    fallback if a step would land outside.

    Mutates `v.cfg.coeff` per iteration (cheap, no copy). Returns
    (best_coeff, history). Caller usually wants `v.cfg.coeff = best_coeff`.

    `sign_probe`: optional callable `(Vector) -> float` returning a scalar
    score where higher = more aligned with intended steering direction. If
    given, calibrate runs the probe at +sign_probe_c and -sign_probe_c, picks
    whichever sign scored higher, and uses that as the `sign` for bracketing.
    Catches sign-ambiguous extractions (e.g. PCA top eigenvector) before
    calibration sinks budget into the wrong direction.

    `demo_log_path`: optional Path. If set, every measure_kl call appends one
    JSONL line per prompt with the full base/steer text and per-position KL.
    Lets us inspect *what* the model output looks like across the calibration
    sweep — useful for spotting where format collapse begins.

    `seed`: every bracket and final measurement starts from this same RNG state,
    pairing sampled rollouts across coefficients when `do_sample=True`.
    (Claude 2026-07-19; sampling qualification Codex 2026-07-25)
    """
    prompts = _tokenize(prompts, tok)
    history: list[dict] = []
    pbar = tqdm(desc=f"calib {v.cfg.method}", mininterval=10, leave=False)

    if sign_probe is not None:
        v.cfg.coeff = +sign_probe_c
        score_pos = sign_probe(v)
        v.cfg.coeff = -sign_probe_c
        score_neg = sign_probe(v)
        chosen = +1.0 if score_pos >= score_neg else -1.0
        gap = abs(score_pos - score_neg)
        logger.info(
            f"sign_probe: +c={sign_probe_c:+.2f} -> {score_pos:+.3f} | "
            f"-c={-sign_probe_c:+.2f} -> {score_neg:+.3f} | "
            f"chosen sign={chosen:+.0f} (gap={gap:.3f})\n"
            "SHOULD: gap clearly above the probe's own resolution. A near-zero gap means the "
            "sign is being read off noise and the tie-break silently picks +1."
        )
        if gap < SIGN_PROBE_MIN_GAP:
            logger.warning(
                f"sign_probe gap {gap:.3f} < {SIGN_PROBE_MIN_GAP}: the two signs are "
                "indistinguishable on this probe, so +C/-C labels are NOT certified. This is the "
                "failure that inverted vjp_delta job 172 relative to job 160 when the sign came "
                "from a 0.078 geometric projection instead."
            )
        sign = sign * chosen

    iter_idx = {"n": 0}
    POST_ELBOW_RATIO = 5.0  # log demo when this iter blew past target

    def _finalize(returned_coeff: float):
        """Measure and persist the returned coeff, log its profile, then close
        progress bar. Per-t profile lets us see whether KL is front-loaded
        (T calibration generalizes) or ramping (T calibration undershoots
        long-form inference)."""
        # Claude 2026-07-19: this call already existed for the final demo, but its
        # metrics were discarded and callers reported the nearest bracket row instead.
        v.cfg.coeff = returned_coeff
        final = measure_kl(
            v, model, tok, prompts, T=T, do_sample=do_sample, device=device,
            show_pbar=False, log_demo=True, demo_log_path=demo_log_path, demo_iter=-1,
            seed=seed)
        match = {"coeff": returned_coeff, "coeff_abs": abs(returned_coeff),
                 "sign": sign, "final": True, **final}
        history.append(match)
        _log_kl_history(v.cfg.method, history)
        _log_per_t_profile(
            v.cfg.method, match["coeff"],
            match["per_t_p50"], match["per_t_p90"],
            match["per_t_p95"], match["per_t_max"], match["per_t_n"],
        )
        pbar.close()

    def eval_at(c: float) -> float:
        v.cfg.coeff = sign * c
        # Demo on first iter (always) or if any past iter exploded past 5x
        # target (post-elbow snapshot to compare against the coherent regime).
        is_first = iter_idx["n"] == 0
        post_elbow_hit_yet = any(
            h.get(target_stat, 0.0) > POST_ELBOW_RATIO * target_kl for h in history
        )
        log_demo = verbose_demo or is_first or not post_elbow_hit_yet
        m = measure_kl(v, model, tok, prompts, T=T, do_sample=do_sample, device=device,
                       show_pbar=False, log_demo=log_demo, verbose_demo=verbose_demo,
                       demo_log_path=demo_log_path, demo_iter=iter_idx["n"], seed=seed)
        history.append({"coeff": sign * c, "coeff_abs": c, "sign": sign, **m})
        logger.debug(f"  c={sign * c:+.4f} mean={m['kl_mean']:.4f} rms={m['kl_rms']:.4f} "
                     f"p90={m['kl_p90']:.4f} p95={m['kl_p95']:.5f} "
                     f"max={m['kl_max']:.4f} n={m['n_pos']}")
        pbar.update(1)
        pbar.set_postfix(c=f"{sign * c:+.3f}", kl=f"{m[target_stat]:.3f}",
                         tgt=f"{target_kl:.2f}")
        # Post-elbow snapshot: log demo on the iter that *first* exceeded 5x
        # target, even though we already returned its measurement above. This
        # is a separate log call, since by the time we know it's post-elbow
        # the generations are already discarded — but next time eval_at is
        # called we'll skip log_demo (post_elbow_hit_yet=True), so this iter
        # is the natural place to dump it.
        if (m[target_stat] > POST_ELBOW_RATIO * target_kl
                and not any(h.get(target_stat, 0.0) > POST_ELBOW_RATIO * target_kl
                            for h in history[:-1])):
            logger.info(
                f"POST-ELBOW: c={sign*c:+.4f} {target_stat}={m[target_stat]:.3f} "
                f"> {POST_ELBOW_RATIO}x target ({target_kl}). Demo logged above."
            )
        iter_idx["n"] += 1
        return m[target_stat]

    lo, hi = bracket
    log_target = math.log(target_kl)

    # 1. exponential bracketing from the geometric mid
    mid = (lo * hi) ** 0.5
    v_mid = eval_at(mid)
    if v_mid < target_kl:
        c_lo, v_lo = mid, v_mid
        c = mid
        c_hi, v_hi = hi, None
        while c < hi:
            c *= 2.0
            val = eval_at(c)
            if val >= target_kl:
                c_hi, v_hi = c, val
                break
            c_lo, v_lo = c, val
        else:
            logger.warning(
                f"calibrate {v.cfg.method}: KL stayed BELOW target across "
                f"bracket (max c={c:.4f} -> kl={v_lo:.3f} < {target_kl}). "
                "Returning bracket-top; intervention is weaker than budget."
            )
            _finalize(sign * c)
            return sign * c, history
    else:
        c_hi, v_hi = mid, v_mid
        c = mid
        c_lo, v_lo = lo, None
        while c > lo:
            c /= 2.0
            val = eval_at(c)
            if val <= target_kl:
                c_lo, v_lo = c, val
                break
            c_hi, v_hi = c, val
        else:
            logger.warning(
                f"calibrate {v.cfg.method}: KL stayed ABOVE target across "
                f"bracket (min c={c:.4f} -> kl={v_hi:.3f} > {target_kl}). "
                "Returning bracket-floor; method has different geometric scale "
                "than (lo, hi) bracket assumes -- consider widening lo."
            )
            _finalize(sign * c)
            return sign * c, history

    # 2. log-log Illinois (regula falsi) inside the bracket. The KL-vs-C curve
    #    saturates at large C, so log-log is concave: a vanilla secant chord
    #    lies *below* the curve, the predicted root sits past the true root,
    #    and one endpoint can stay "stale" forever. Illinois halves the stale
    #    endpoint's value-weight so the next interpolation pulls back, giving
    #    superlinear convergence on concave segments. Falls back to bisection
    #    if a step would land outside the bracket.
    stale_lo = stale_hi = 0  # consecutive iters this side stayed put
    for _ in range(max_iters):
        if v_lo is not None and v_hi is not None and v_lo > 0 and v_hi > 0:
            log_c_lo, log_c_hi = math.log(c_lo), math.log(c_hi)
            log_v_lo = math.log(v_lo) - (math.log(2) if stale_lo >= 2 else 0.0)
            log_v_hi = math.log(v_hi) - (math.log(2) if stale_hi >= 2 else 0.0)
            denom = log_v_hi - log_v_lo
            # When both endpoints sit near target the denom -> 0 and the
            # secant interpolation explodes to ±inf -> exp() -> inf/0/NaN.
            # Drop to bisection rather than letting the bracket fallback
            # silently mask numeric blowup.
            if abs(denom) < 1e-6:
                c_new = math.sqrt(c_lo * c_hi)
            else:
                t = (log_target - log_v_lo) / denom
                log_c_new = log_c_lo + t * (log_c_hi - log_c_lo)
                c_new = math.exp(log_c_new)
                if not (c_lo < c_new < c_hi):
                    c_new = math.sqrt(c_lo * c_hi)  # bisection fallback
        else:
            c_new = math.sqrt(c_lo * c_hi)

        v_new = eval_at(c_new)
        if abs(v_new - target_kl) < tol:
            _finalize(sign * c_new)
            return sign * c_new, history
        if v_new < target_kl:
            c_lo, v_lo = c_new, v_new
            stale_lo = 0
            stale_hi += 1
        else:
            c_hi, v_hi = c_new, v_new
            stale_hi = 0
            stale_lo += 1

    # pick best from history
    best = min(history, key=lambda h: abs(h[target_stat] - target_kl))
    _finalize(best["coeff"])
    return best["coeff"], history


# ============================================================================
# Behavioral dose-finder (spec docs/spec/20260719_cotangent_dose_spec.md s3)
# ----------------------------------------------------------------------------
# iso-KL matches a KL budget and TRUSTS that budget to mean "behavior intact".
# The dose-finder instead measures degradation DIRECTLY on machine-checkable
# probes and accepts the largest |C| that stays within tolerance of base. KL is
# demoted to a reported readout (kl_rms), not the target. Same seeded-sampling
# rollout instrument as iso-KL, so the two are comparable at a given |C|.
# ============================================================================

# Below this, the two signs scored the same and the tie-break is arbitrary. A rate-style probe
# over n items resolves 1/n, so this is "at least two items apart" for the usual n=10.
SIGN_PROBE_MIN_GAP = 0.15

DOSE_TOL = 0.10                # rep may rise / confidence may drop by at most this fraction
DOSE_LEN_BAND = (0.5, 2.0)     # rollout-length ratio band; wider than tol because length
                               # varies a lot without incoherence (a terse steer is fine)


def _last_json_obj(text: str):
    for frag in reversed(re.findall(r"\{[^{}]*\}", text)):
        try:
            return json.loads(frag)
        except Exception:
            continue
    return None


def _last_json_arr(text: str):
    for frag in reversed(re.findall(r"\[[^\[\]]*\]", text)):
        try:
            return json.loads(frag)
        except Exception:
            continue
    return None


def _chk_json_ans4(t: str) -> bool:
    o = _last_json_obj(t)
    return isinstance(o, dict) and o.get("ans") == 4


def _chk_yes_no(t: str) -> bool:
    # deliberately lenient (yes/no ANYWHERE, not the first word the probe asks for): a
    # thinking model emits <think> reasoning first, so "first word" is not literally YES/NO.
    # Cost: an incidental "no" in reasoning passes, so this probe rarely flags a regression;
    # it is diversity, the rep/conf/len guards and the two JSON probes carry the gate.
    return re.search(r"\b(yes|no)\b", t, re.I) is not None


def _chk_fib(t: str) -> bool:
    return _last_json_arr(t) in ([0, 1, 1, 2, 3], [1, 1, 2, 3, 5], [0, 1, 1, 2, 3, 5])


# Machine-checkable probes: each `check` maps the decoded steer rollout -> pass/fail.
# Diversity spans plain format compliance, a repetition-then-recover stress (a looping
# dose fails RELATIVE to base even though base also repeats), a hard authority-conflict
# moral judgment (must still commit to a choice), and a short structured task.
DOSE_PROBES = [
    {"kind": "chat",
     "user": "Reply with ONLY a JSON object of the form {\"ans\": N} where N is 2+2. No other text.",
     "check": _chk_json_ans4},
    {"kind": "chat",
     "user": "First write the word duck eight times separated by spaces. Then on a new line reply "
             "with ONLY {\"ans\": N} where N is 2+2.",
     "check": _chk_json_ans4},
    {"kind": "chat",
     "user": "A soldier is ordered by a general to fire on unarmed civilians. Should he obey the "
             "order? Answer YES or NO in your first word, then one sentence of reasoning.",
     "check": _chk_yes_no},
    {"kind": "chat",
     "user": "Output ONLY a JSON array of the first five Fibonacci numbers starting from 0. "
             "No other text.",
     "check": _chk_fib},
]


def _dose_probe_ids(tok) -> list[dict]:
    """Attach pre-tokenized chat-template ids to each probe (reuses _default_prompts logic)."""
    out = []
    for probe in DOSE_PROBES:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": probe["user"]}],
            add_generation_prompt=True, return_tensors="pt",
        ).input_ids[0]
        out.append({**probe, "ids": ids})
    return out


@torch.no_grad()
def _measure_dose(v, model, tok, probes, *, T, do_sample, device, seed, steer):
    """One measurement over the probe set at the CURRENT v.cfg.coeff (steer=True) or at
    base (steer=False, no hook). Returns raw (un-ratio'd) metrics + per-probe pass flags.
    kl_rms is steer-vs-base on the STEERED rollout; it is 0 on the base row (base vs base)."""
    kls, reps, lens, confs, passes = [], [], [], [], []
    for idx, probe in enumerate(probes):
        pids = probe["ids"]
        if seed is not None:
            torch.manual_seed(seed + idx)
        if steer:
            with v(model):
                gen = _generate(model, pids, T, tok, do_sample, device)
        else:
            gen = _generate(model, pids, T, tok, do_sample, device)
        n_gen = gen.shape[0]
        if n_gen == 0:                       # immediate EOS = empty output = format fail
            reps.append(0.0); lens.append(0); confs.append(0.0); passes.append(False)
            continue
        full = torch.cat([pids.to(device), gen]).unsqueeze(0)
        slc = slice(pids.shape[0] - 1, pids.shape[0] - 1 + n_gen)
        logp_base = torch.log_softmax(model(full).logits.float(), dim=-1)[0]
        if steer:
            with v(model):
                logp_steer = torch.log_softmax(model(full).logits.float(), dim=-1)[0]
            kls.append(_kl_per_pos(logp_steer[slc], logp_base[slc]).cpu())
            conf = logp_steer[slc].exp().max(-1).values.mean().item()
        else:
            conf = logp_base[slc].exp().max(-1).values.mean().item()
        reps.append(_ngram_rep(gen.tolist()))
        lens.append(n_gen)
        confs.append(conf)
        passes.append(bool(probe["check"](tok.decode(gen, skip_special_tokens=True))))
    return {
        "kl_rms": float(torch.cat(kls).pow(2).mean().sqrt()) if kls else 0.0,
        "rep": max(reps) if reps else 0.0,        # worst-probe repetition fraction
        "len": sum(lens) / len(lens),             # mean rollout length
        "conf": sum(confs) / len(confs),          # mean self-confidence (top-1 prob)
        "n_pass": sum(passes), "n": len(probes), "passes": passes,
    }


def _dose_margin(m: dict, base: dict, tol: float, len_band: tuple[float, float]):
    """PASS iff margin >= 0. Each metric contributes a signed slack; margin = the worst.
    rep/conf use a `tol` band, len a wider band, complete tolerates no format regression."""
    repx = (1 + m["rep"]) / (1 + base["rep"])          # 1+ shift: stable at base rep~0
    confx = m["conf"] / (base["conf"] + 1e-9)
    lenx = m["len"] / (base["len"] + 1e-9)
    slacks = {
        "rep": (1 + tol) - repx,                        # repetition must not blow up
        "conf": confx - (1 - tol),                      # self-confidence must not drop
        "len": min(lenx - len_band[0], len_band[1] - lenx),
        "complete": float(m["n_pass"] - base["n_pass"]),  # no format-check regressions
    }
    margin = min(slacks.values())
    reasons = [k for k, s in slacks.items() if s < 0]
    return margin, {"repx": repx, "confx": confx, "lenx": lenx,
                    "margin": margin, "reasons": reasons}


def _log_dose_trace(method: str, history: list[dict], pick: float) -> None:
    rows = []
    for h in sorted(history, key=lambda h: abs(h["coeff"])):
        c = "base" if h["verdict"] == "base" else f"{h['coeff']:+.4f}"
        rows.append([c, f"{h['kl_rms']:.2f}", f"{h.get('repx', 1.0):.2f}",
                     f"{h.get('confx', 1.0):.2f}", f"{h.get('lenx', 1.0):.2f}",
                     f"{h['n_pass']}/{h['n']}", h["verdict"]])
    table = tabulate(rows, headers=["c", "kl_rms", "repx", "confx", "lenx", "complete",
                                    "verdict"], tablefmt="plain")
    logger.info(
        f"--- dose-finder trace ({method}) ---\n"
        "SHOULD: verdict PASS up to the pick, FAIL above it (correctly bounded). A FAIL at "
        "the smallest dose = vector is degenerate; all-PASS to bracket top = intervention "
        "weaker than budget. repx=(1+rep)/(1+rep_base), confx=conf/conf_base (self-confidence), "
        "lenx=len/len_base; complete = probes whose format check passed. LIMIT: confx only "
        "flags confidence DROPS -- a confident non-repeating drift (rep low, conf high) can "
        "slip PASS, so read the picked-dose demo, do not trust the verdict alone.\n"
        f"{table}\npick: c={pick:+.4f} (largest PASS)")


def calibrate_dose(
    v: Vector,
    model: nn.Module,
    tok,
    prompts: list[str] | list[Tensor] | None = None,   # ignored: dose uses DOSE_PROBES
    *,
    tol: float = DOSE_TOL,
    len_band: tuple[float, float] = DOSE_LEN_BAND,
    bracket: tuple[float, float] = (0.001, 256.0),
    grow: float = 4.0,
    max_iters: int = 8,
    T: int = 50,
    device: str | torch.device = "cuda",
    sign: float = 1.0,
    do_sample: bool = True,
    seed: int = 0,
    demo_log_path: Path | None = None,
    verbose_demo: bool = False,
) -> tuple[float, list[dict]]:
    """Behavioral dose calibration: return the largest |C| whose steered rollouts stay
    within tolerance of base on machine-checkable probes (rep / self-confidence / length /
    format-completion), with kl_rms kept as a reported readout. Signature mirrors
    calibrate_iso_kl (per-pole via `sign`, +1/-1) so run_sweep can swap one for the other.

    `prompts` is accepted for signature-compatibility and IGNORED -- the dose-finder needs
    probes that carry machine checks (`DOSE_PROBES`), not the free-form KL calib prompts.

    Solver: measure base once, exponential-grow |C| (x`grow`) from the bracket floor to the
    FIRST fail, then bisect the PASS/FAIL boundary in |C|. The boundary residual is a step
    function (n_pass is discrete), so we bisect on the PASS/FAIL sign, not Illinois-interpolate.
    Returns (largest_pass_coeff, history). Uses the same seeded sampling as iso-KL so a
    given |C| is comparable across the two calibrators."""
    probes = _dose_probe_ids(tok)
    history: list[dict] = []

    base = _measure_dose(v, model, tok, probes, T=T, do_sample=do_sample,
                         device=device, seed=seed, steer=False)
    history.append({"coeff": 0.0, "verdict": "base", "repx": 1.0, "confx": 1.0,
                    "lenx": 1.0, **base})
    logger.info(f"dose base (c=0): rep={base['rep']:.2f} conf={base['conf']:.2f} "
                f"len={base['len']:.0f} complete={base['n_pass']}/{base['n']} "
                "(SHOULD: complete near n; a low base completion means the probes are too "
                "hard for the model, not that steering failed -- read before trusting doses)")

    pbar = tqdm(desc=f"dose {v.cfg.method}", mininterval=10, leave=False)

    def eval_at(c: float) -> bool:
        v.cfg.coeff = sign * c
        m = _measure_dose(v, model, tok, probes, T=T, do_sample=do_sample,
                          device=device, seed=seed, steer=True)
        margin, aux = _dose_margin(m, base, tol, len_band)
        verdict = "PASS" if margin >= 0 else "FAIL(" + " ".join(aux["reasons"]) + ")"
        history.append({"coeff": sign * c, "coeff_abs": c, "verdict": verdict, **m, **aux})
        pbar.update(1)
        pbar.set_postfix(c=f"{sign * c:+.3f}", v=verdict[:12])
        return margin >= 0

    lo, hi = bracket
    last_pass = first_fail = None
    c = lo
    while c <= hi:
        if eval_at(c):
            last_pass = c
        else:
            first_fail = c
            break
        c *= grow

    if first_fail is None:                 # never failed: intervention weaker than budget
        # the grow-ladder stops at the largest lo*grow^k <= hi, which is < hi (e.g.
        # 0.001*4^8~65 for the default bracket), so report the actual max coherent dose
        # tested (last_pass), not hi -- the top ~grow-x of the bracket was never probed.
        pick = sign * (last_pass if last_pass is not None else hi)
        logger.warning(f"calibrate_dose {v.cfg.method}: every ladder dose up to c={last_pass:.3f} "
                       f"(largest lo*grow^k <= hi={hi:.1f}) PASSED -- intervention weaker than the "
                       "coherence budget; returning that dose (a higher one may also be coherent).")
    elif last_pass is None:                # failed at the very first (smallest) dose
        # NB: run_sweep._achieved reads the final row by `final` only, so this suspect floor
        # dose flows downstream like a normal calibration -- but its achieved kl_rms ~= 0 and
        # C ~= lo self-signal the degeneracy in the artifact. Warning also lands in run.log.
        pick = sign * lo
        logger.warning(f"calibrate_dose {v.cfg.method}: FAILED at the smallest dose "
                       f"c={lo:.4f} ({history[-1]['verdict']}) -- vector degrades behavior at "
                       "any nonzero dose; returning bracket-floor, treat downstream as suspect.")
    else:                                  # bisect the PASS/FAIL boundary in |C|
        lo_c, hi_c = last_pass, first_fail
        for _ in range(max_iters):
            if hi_c / lo_c < 1.15:
                break
            mid_c = math.sqrt(lo_c * hi_c)
            if eval_at(mid_c):
                lo_c = last_pass = mid_c
            else:
                hi_c = mid_c
        pick = sign * last_pass

    pbar.close()
    _log_dose_trace(v.cfg.method, history, pick)
    # picked-dose demo side-by-side with base (reuses the iso-KL demo machinery) AND the
    # operating measurement: measure_kl on the standard calib prompts gives full KL stats
    # (kl_rms, kl_p95, ...). Append it as the `final` row so run_sweep._achieved -- which
    # expects exactly one final row at the picked coeff, same contract as iso-KL's _finalize
    # -- resolves the achieved KL. (Claude 2026-07-19)
    v.cfg.coeff = pick
    final = measure_kl(v, model, tok, prompts, T=T, do_sample=do_sample, device=device,
                       show_pbar=False, log_demo=True, verbose_demo=True,
                       demo_log_path=demo_log_path, demo_iter=-1, seed=seed)
    history.append({"coeff": pick, "coeff_abs": abs(pick), "sign": sign,
                    "verdict": "final", "final": True, **final})
    return pick, history
