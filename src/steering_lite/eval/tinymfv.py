"""Thin adapter over the `tinymfv` package (forced-choice MFT vignettes).

Translates the new `tinymfv.evaluate` return shape into the per-foundation
log-ratio dicts the rest of steering_lite expects (multibool-shaped report).

    from steering_lite.eval.tinymfv import evaluate_with_vector

    v = sl.train(model, tok, pos, neg, sl.MeanDiffC(layers=(15,)))
    with v(model, C=2.0):
        report = evaluate_with_vector(model, tok, name="classic")
    print(report["table"])

Forced-choice format is structurally enforced (prefill = `... "violation": "`),
so the legacy soft `pmass` quantity is meaningless. The new OOD signal is
`mean_margin` in nats: healthy model ~1-3 nats, destroyed model -> 0.
`raw_pmass` is retained for downstream compatibility but populated with
margins (per row, broadcast across foundations) so caller code that thresholds
on it still gets a real OOD signal.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from loguru import logger

from tinymfv import evaluate as _tinymfv_evaluate
from tinymfv.guided import (
    _DEFAULT_FORCED_FOUNDATIONS,
    guided_rollout_forced_choice,
)


def _per_row_to_multibool(per_row: list[dict]) -> tuple[
    dict[str, dict[str, float]], dict[str, float], list[float], list[float]
]:
    """Translate per_row 7-vec p into multibool-shaped (raw_logratios, raw_pmass).

    raw_logratios[vid|cond][f_lower] = logit(p[f])
    raw_pmass[vid|cond|f_lower] = margin (broadcast; all foundations share it
    since forced-choice gives a single margin per row).
    """
    raw_logratios: dict[str, dict[str, float]] = {}
    raw_pmass: dict[str, float] = {}
    margins: list[float] = []
    p_social: list[float] = []
    foundations = list(_DEFAULT_FORCED_FOUNDATIONS)
    soc_idx = foundations.index("social")
    for r in per_row:
        key = f"{r['id']}|{r['condition']}"
        d: dict[str, float] = {}
        for fi, fname in enumerate(foundations):
            p_f = float(r["p"][fi])
            p_clip = max(1e-3, min(1 - 1e-3, p_f))
            d[fname] = math.log(p_clip / (1.0 - p_clip))
        raw_logratios[key] = d
        m = float(r["margin"])
        margins.append(m)
        p_social.append(float(r["p"][soc_idx]))
        for fname in foundations:
            raw_pmass[f"{key}|{fname}"] = m
    return raw_logratios, raw_pmass, margins, p_social


def _log_eval_demo_trace(model, tok, name: str, max_think_tokens: int,
                         vector=None, log_demo: bool = True) -> None:
    """One real forced-choice rollout on the first Authority vignette.

    vector=None: bare trace. Vector given: shows top1 and margin under the
    currently-attached steering. Caller controls attachment; we never detach.
    """
    if not log_demo:
        return
    from tinymfv.data import load_vignettes

    vignettes = load_vignettes(name)
    if not vignettes:
        logger.warning(f"tinymfv: no vignettes for name={name!r}, skipping demo trace")
        return
    auth_vigs = [v for v in vignettes if v.get("foundation_coarse") == "Authority"]
    r = auth_vigs[0] if auth_vigs else vignettes[0]
    cond = "other_violate"
    method_name = getattr(getattr(vector, "cfg", None), "method", "none")
    coeff = getattr(getattr(vector, "cfg", None), "coeff", 0.0)
    label = "BARE" if vector is None else f"STEERED ({method_name}, c={coeff:+.4f})"
    user_prompt = r[cond]

    with torch.no_grad():
        results = guided_rollout_forced_choice(
            model, tok, [user_prompt],
            max_think_tokens=max_think_tokens,
            verbose=True,
        )
    res = results[0]
    p_sorted = sorted(res.p.items(), key=lambda kv: -kv[1])
    p_str = "  ".join(f"{f}={p:.2f}" for f, p in p_sorted[:4])
    logger.info(
        "[demo] SHOULD: top1 matches vignette foundation; margin >= 0.3 nats.\n"
        f"=== EVAL demo  stage=eval  method={method_name}  vid={r.get('id','?')}  "
        f"cond={cond}  max_think={max_think_tokens} ===\n"
        f"  {label}: top1={res.top1}  margin={res.margin:+.3f}nat  "
        f"label={r.get('foundation_coarse','?')}\n"
        f"  p: {p_str}\n"
        f"=== /EVAL ==="
    )


def evaluate_with_vector(
    model, tok, *,
    name: str = "classic",
    max_think_tokens: int = 64,
    vector=None,
    log_demo: bool = True,
    vignettes: list[dict] | None = None,
    batch_size: int = 8,
    **kwargs,
) -> dict:
    """Run forced-choice eval with whatever steering is currently attached.

    Returns a multibool-shaped report:
      - raw_logratios: {vid|cond: {f_lower: logit(p[f])}}  (7 foundations incl. social)
      - raw_pmass:     {vid|cond|f: margin}                (margin in nats; OOD proxy)
      - wrongness:     mean (1 - p[social]) across rows  ∈ [0,1]
      - mean_margin:   mean margin over rows (nats)        -- main OOD signal
      - mean_js:       JS vs label dist (nats)             (None if no labels)
      - top1_acc:      argmax-match vs label argmax        (None if no labels)
      - table:         per-foundation pandas DataFrame from tinymfv
      - info:          diagnostics dict
      - per_row:       list of per-row dicts (id, condition, p, label, top1, margin)
    """
    _log_eval_demo_trace(model, tok, name=name, max_think_tokens=max_think_tokens,
                         vector=vector, log_demo=log_demo)
    rep = _tinymfv_evaluate(
        model, tok, name=name, vignettes=vignettes,
        max_think_tokens=max_think_tokens, batch_size=batch_size,
        return_per_row=True, **kwargs,
    )
    raw_logratios, raw_pmass, margins, p_social = _per_row_to_multibool(rep["per_row"])
    mean_margin = float(np.mean(margins)) if margins else float("nan")
    wrongness = float(np.mean([1.0 - ps for ps in p_social])) if p_social else float("nan")
    return {
        "raw_logratios": raw_logratios,
        "raw_pmass": raw_pmass,
        "wrongness": wrongness,
        "mean_margin": mean_margin,
        "mean_js": rep["mean_js"],
        "top1_acc": rep["top1_acc"],
        "table": rep["table"],
        "info": rep["info"],
        "per_row": rep["per_row"],
    }


# Alias: sweep script imports this name; forced-choice replaces old multibool.
evaluate_multibool = evaluate_with_vector
