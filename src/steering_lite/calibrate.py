"""Iso-KL calibration: find the coeff so KL(steer || base) hits a target.

Two functions:
- `measure_kl(...)`: roll out T tokens with steering attached, then teacher-force
  re-score base + steer over the rolled-out sequence, return per-token KL stats.
- `calibrate_iso_kl(...)`: log-log secant solver. Brackets exponentially, then
  interpolates in `(log C, log stat)` space. KL(p_C || p_0) ~ C^2 in the small-C
  regime, so log-log slope is ~2 and secant converges in ~3-5 iters (vs ~10 for
  pure bisection).

Recommended: greedy decode with target_kl=1.0 nat, target_stat="kl_p95".
Greedy makes the statistic deterministic so the solver doesn't fight noise.
"""
from __future__ import annotations
import json
import math
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
    rows = [
        [str(i), f"{h['coeff']:+.4f}", f"{h['kl_mean']:.4f}", f"{h['kl_p90']:.4f}",
         f"{h['kl_p95']:.5f}", f"{h['kl_max']:.4f}", str(h['n_pos'])]
        for i, h in indexed
    ]
    table = tabulate(rows, headers=["i", "c", "mean", "p90", "p95", "max", "n"], tablefmt="plain")
    logger.info(f"\n--- iso-KL bracket trace ({method}, {len(history)} iters) ---\n{table}")


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
) -> dict:
    """Roll out T tokens with steering attached, then score under base
    (detached) and steer (re-attached). Returns KL summary stats and per-token
    KL distributions across positions.

    Logging knobs (method/eval-agnostic — only inputs and outputs):
      log_demo       : emit a BASE-vs-STEER comparison for prompt 0 to loguru.
      demo_log_path  : if set, append one JSONL line per (iter, prompt) with
                       full base text, steer text, and per-position KL.
      demo_iter      : iteration index, written into JSONL records (cosmetic).
    """
    prompts = _tokenize(prompts, tok)
    all_kls = []
    per_t = [[] for _ in range(T)]
    need_base_gen = log_demo or demo_log_path is not None

    for idx, pids in enumerate(tqdm(prompts, desc="measure_kl",
                                    mininterval=60, disable=not show_pbar)):
        with v(model):
            gen = _generate(model, pids, T, tok, do_sample, device)
        n_gen = gen.shape[0]
        if n_gen == 0:
            continue
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

        # Demo: extra base-only gen for side-by-side text. One per measure_kl
        # call (idx==0) for stdout, all prompts for JSONL.
        if need_base_gen:
            base_gen = _generate(model, pids, T, tok, do_sample, device)
            base_full = torch.cat([pids.to(device), base_gen])
            decoded_base = tok.decode(base_full, skip_special_tokens=False)
            decoded_steer = tok.decode(full_ids, skip_special_tokens=False)
            if log_demo and idx == 0:
                logger.info(
                    f"EXPECT: same prompt under c=0 vs c={v.cfg.coeff:+.4f}; both coherent; "
                    "steered should differ from base but not collapse.\n"
                    f"\n=== CALIBRATE demo trace (T={T}, kl_p95_so_far) ===\n"
                    f"--- BASE (c=0) ---\n{decoded_base}\n"
                    f"\n--- STEER (c={v.cfg.coeff:+.4f}) ---\n{decoded_steer}\n"
                    f"=== /CALIBRATE ==="
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
        "kl_p50": float(cat.quantile(0.50)),
        "kl_p90": float(cat.quantile(0.90)),
        "kl_p95": float(cat.quantile(0.95)),
        "kl_max": float(cat.max()),
        "n_pos": int(cat.numel()),
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
    target_kl: float = 0.5,
    target_stat: str = "kl_p95",
    bracket: tuple[float, float] = (0.001, 256.0),
    tol: float = 0.05,
    max_iters: int = 12,
    T: int = 50,
    device: str | torch.device = "cuda",
    sign: float = 1.0,
    sign_probe: Callable[[Vector], float] | None = None,
    sign_probe_c: float = 1.0,
    demo_log_path: Path | None = None,
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
        logger.info(
            f"sign_probe: +c={sign_probe_c:+.2f} -> {score_pos:+.3f} | "
            f"-c={-sign_probe_c:+.2f} -> {score_neg:+.3f} | "
            f"chosen sign={chosen:+.0f} (gap={abs(score_pos - score_neg):.3f})"
        )
        sign = sign * chosen

    iter_idx = {"n": 0}
    POST_ELBOW_RATIO = 5.0  # log demo when this iter blew past target

    def _finalize(returned_coeff: float):
        """Log bracket trace + per-t profile for the returned coeff, close
        progress bar. Per-t profile lets us see whether KL is front-loaded
        (T calibration generalizes) or ramping (T calibration undershoots
        long-form inference)."""
        _log_kl_history(v.cfg.method, history)
        match = next((h for h in history if abs(h["coeff"] - returned_coeff) < 1e-9), None)
        if match is None and history:
            match = min(history, key=lambda h: abs(h["coeff"] - returned_coeff))
        if match is not None and "per_t_p95" in match:
            _log_per_t_profile(
                v.cfg.method, match["coeff"],
                match["per_t_p50"], match["per_t_p90"],
                match["per_t_p95"], match["per_t_max"], match["per_t_n"],
            )
        # Demo at the calibrated coeff (1x target) -- the most useful snapshot.
        # The post-elbow demo above shows collapse; this shows the operating point.
        v.cfg.coeff = returned_coeff
        measure_kl(v, model, tok, prompts, T=T, do_sample=True, device=device,
                   show_pbar=False, log_demo=True, demo_log_path=demo_log_path,
                   demo_iter=-1)
        pbar.close()

    def eval_at(c: float) -> float:
        v.cfg.coeff = sign * c
        # Demo on first iter (always) or if any past iter exploded past 5x
        # target (post-elbow snapshot to compare against the coherent regime).
        is_first = iter_idx["n"] == 0
        post_elbow_hit_yet = any(
            h.get(target_stat, 0.0) > POST_ELBOW_RATIO * target_kl for h in history
        )
        log_demo = is_first or not post_elbow_hit_yet
        m = measure_kl(v, model, tok, prompts, T=T, do_sample=True, device=device,
                       show_pbar=False, log_demo=log_demo,
                       demo_log_path=demo_log_path, demo_iter=iter_idx["n"])
        history.append({"coeff": sign * c, "coeff_abs": c, "sign": sign, **m})
        logger.debug(f"  c={sign * c:+.4f} mean={m['kl_mean']:.4f} "
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
                f"POST-ELBOW: c={sign*c:+.4f} kl_{target_stat}={m[target_stat]:.3f} "
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
