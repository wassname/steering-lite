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
import math
from typing import Callable

import torch
from loguru import logger
from torch import Tensor
from torch import nn
from tqdm.auto import tqdm

from .config import SteeringConfig
from .vector import Vector


_demo_logged = {"flag": False}


# Generic user messages for cheap default calibration. Diversity > length:
# 4 domains spanning factual / narrative / technical / introspective.
# Wrapped through tok.apply_chat_template so calibration reflects real chat use.
DEFAULT_MESSAGES = [
    "The eiffel tower is in Paris",
    "埃菲尔铁塔🗼位于天都城",
    "Tell me a greentext story about a small village during the smaller Martion carrot bubble.",
    "Think step by step to calculate the integral of x^2 from 0 to 1 in lean4.",
]


def _tokenize(prompts: list[str] | list[Tensor] | None, tok) -> list[Tensor]:
    if prompts is None:
        prompts = DEFAULT_MESSAGES
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
) -> dict:
    """Roll out T tokens with steering attached, then score under base
    (detached) and steer (re-attached). Returns KL summary stats.
    """
    prompts = _tokenize(prompts, tok)
    all_kls = []
    per_t = [[] for _ in range(T)]

    for idx, pids in enumerate(tqdm(prompts, desc="measure_kl", mininterval=60)):
        with v(model):
            gen = _generate(model, pids, T, tok, do_sample, device)
        n_gen = gen.shape[0]
        if n_gen == 0:
            continue
        full_ids = torch.cat([pids.to(device), gen])
        if idx == 0 and not _demo_logged["flag"]:
            _demo_logged["flag"] = True
            base_gen = _generate(model, pids, T, tok, do_sample, device)
            base_full = torch.cat([pids.to(device), base_gen])
            decoded_base = tok.decode(base_full, skip_special_tokens=False)
            decoded_steer = tok.decode(full_ids, skip_special_tokens=False)
            logger.info(
                f"EXPECT: same prompt under c=0 vs c={v.cfg.coeff:+.4f}; both coherent; "
                "steered should differ from base but not collapse.\n"
                f"\n=== CALIBRATE demo trace (T={T}) ===\n"
                f"--- BASE (c=0) ---\n{decoded_base}\n"
                f"\n--- STEER (c={v.cfg.coeff:+.4f}) ---\n{decoded_steer}\n"
                f"=== /CALIBRATE ==="
            )
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

    cat = torch.cat(all_kls)
    return {
        "kl_mean": float(cat.mean()),
        "kl_p50": float(cat.quantile(0.50)),
        "kl_p90": float(cat.quantile(0.90)),
        "kl_p95": float(cat.quantile(0.95)),
        "kl_max": float(cat.max()),
        "n_pos": int(cat.numel()),
        "per_t_mean": [sum(xs) / len(xs) if xs else 0.0 for xs in per_t],
        "per_t_max": [max(xs) if xs else 0.0 for xs in per_t],
    }


def calibrate_iso_kl(
    v: Vector,
    model: nn.Module,
    tok,
    prompts: list[str] | list[Tensor] | None = None,
    *,
    target_kl: float = 1.0,
    target_stat: str = "kl_p95",
    bracket: tuple[float, float] = (0.01, 16.0),
    tol: float = 0.05,
    max_iters: int = 12,
    T: int = 20,
    device: str | torch.device = "cuda",
    sign: float = 1.0,
    sign_probe: Callable[[Vector], float] | None = None,
    sign_probe_c: float = 1.0,
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
    """
    _demo_logged["flag"] = False
    prompts = _tokenize(prompts, tok)
    history: list[dict] = []

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

    def eval_at(c: float) -> float:
        v.cfg.coeff = sign * c
        m = measure_kl(v, model, tok, prompts, T=T, do_sample=False, device=device)
        history.append({"coeff": sign * c, "coeff_abs": c, "sign": sign,
                        **{k: val for k, val in m.items() if k not in ("per_t_mean", "per_t_max")}})
        logger.info(f"  c={sign * c:+.4f} mean={m['kl_mean']:.3f} "
                    f"p50={m['kl_p50']:.3f} p90={m['kl_p90']:.3f} "
                    f"p95={m['kl_p95']:.3f} max={m['kl_max']:.3f} n={m['n_pos']}")
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
            return sign * c, history

    # 2. log-log Illinois (regula falsi) inside the bracket. The KL-vs-C curve
    #    saturates at large C, so log-log is concave: a vanilla secant chord
    #    lies *below* the curve, the predicted root sits past the true root,
    #    and one endpoint can stay "stale" forever. Illinois halves the stale
    #    endpoint's value-weight so the next interpolation pulls back, giving
    #    superlinear convergence on concave segments. Falls back to bisection
    #    if a step would land outside the bracket.
    stale_lo = stale_hi = 0  # consecutive iters this side stayed put
    for _ in tqdm(range(max_iters), desc=f"calib {v.cfg.method}", mininterval=60, leave=False):
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
    return best["coeff"], history
