"""Thin adapter over the `tinymfv` package (tiny moral-foundations vignettes).

Lets you run a steering vector through the tinymfv `evaluate` loop without
copying the eval into this repo. Use as:

    from steering_lite.eval.tinymfv import evaluate_with_vector

    v = sl.train(model, tok, pos, neg, sl.MeanDiffC(layers=(15,)))
    with v(model, C=2.0):
        report = evaluate_with_vector(model, tok, name="scifi")
    print(report["table"])

Requires `tinymfv` to be importable (sibling repo `tiny-mcf-vignettes`).
"""
from __future__ import annotations

import torch
from loguru import logger

# --- think-briefly patch -----------------------------------------------------
# Prepend "Think briefly. " to the JSON instruction in tinymfv.FRAMES so the
# CoT budget is short enough to reach the JSON answer within max_think_tokens.
# Must live exactly once and apply to every eval path (sweep, baselines, ad-hoc
# notebooks) -- doing it via FRAMES is the single source of truth that
# tinymfv.eval and our demo wrappers both read at runtime.
def _patch_think_briefly() -> None:
    from tinymfv.core import FRAMES
    PREFIX = "Think briefly. "
    for fname, fr in FRAMES.items():
        if not fr["q"].startswith(PREFIX):
            fr["q"] = PREFIX + fr["q"]

_patch_think_briefly()
del _patch_think_briefly


@torch.no_grad()
def _demo_via_guided(model, tok, user_prompt: str, frame, max_think_tokens: int,
                     verbose: bool = False):
    """Run tinymfv's guided_rollout (same path eval uses). Returns GuidedResult
    with raw_full_text, pmass_format, logratio_ab, p_true.

    verbose=True triggers guided_rollout to log the full prefix+suffix (with
    special tokens, no skip) plus a free-form continuation -- the actual raw
    trace the model sees at scoring time."""
    from tinymfv.guided import guided_rollout, choice_token_ids_tf
    return guided_rollout(
        model, tok,
        user_prompt=user_prompt,
        choice_token_ids=choice_token_ids_tf(tok),
        max_think_tokens=max_think_tokens,
        schema_hint=frame["q"],
        prefill=frame["prefill"],
        verbose=verbose,
    )


def _fmt_trace(label: str, res, tok=None, schema_hint: str | None = None) -> str:
    # Reconstruct exactly what guided_rollout fed to model.generate (phase1):
    # apply_chat_template([{user: up + schema_hint}], add_generation_prompt=True) + "<think>\n" + think + "</think>"
    raw = ""
    if tok is not None:
        try:
            user_content = (f"{res.user_prompt}\n\n{schema_hint}"
                            if schema_hint else res.user_prompt)
            chat = tok.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False, add_generation_prompt=True,
            )
            raw = f"{chat}<think>\n{res.think_text}</think>"
        except Exception as e:  # tokenizer w/o chat template -- fall back
            raw = f"<chat_template_unavailable: {e}>"
    body = (
        f"RAW CHATML (what model saw + generated, no skip-special):\n{raw}\n"
        f"---\n"
        f"USER (extracted):\n{res.user_prompt}\n"
        f"---\n"
        f"THINK (extracted):\n{res.think_text}"
    )
    return (
        f"  --- {label} ---\n"
        f"  pmass={res.pmass_format:.3f}  logratio={res.logratio_ab:+.3f}  p_true={res.p_true:.3f}\n"
        + "\n".join("  " + l for l in body.splitlines())
    )


def _log_eval_demo_trace(model, tok, name: str, max_think_tokens: int,
                         vector=None, log_demo: bool = True) -> None:
    """One real guided_rollout on the first Authority vignette -- same code path as eval.

    vector=None: bare trace only.
    vector given: BASE / +C / -C shown together with pmass+logratio per condition.
    log_demo=False: skip (used when the paired demo was already emitted).
    """
    if not log_demo:
        return
    from tinymfv.data import load_vignettes
    from tinymfv.core import CONDITIONS, FRAMES

    vignettes = load_vignettes(name)
    if not vignettes:
        logger.warning(f"tinymfv: no vignettes for name={name!r}, skipping demo trace")
        return
    auth_vigs = [v for v in vignettes if v.get("foundation") == "Authority"]
    r = auth_vigs[0] if auth_vigs else vignettes[0]
    cond = next(iter(CONDITIONS))
    frame_name, frame = next(iter(FRAMES.items()))
    user_prompt = r[cond]
    method_name = getattr(getattr(vector, 'cfg', None), 'method', 'none')
    header = (f"stage=eval  method={method_name}  "
              f"vignette={r.get('id','?')}  cond={cond}  max_think={max_think_tokens}")

    # Run demo using whatever steering is already attached on the model.
    # Caller controls attachment; this function MUST NOT detach/reattach
    # because that destroys the outer `with vector(model):` context.
    res = _demo_via_guided(model, tok, user_prompt, frame, max_think_tokens, verbose=True)
    label = "BARE" if vector is None else f"STEERED (method={method_name}, c={vector.cfg.coeff:+.4f})"
    logger.info(
        "EXPECT: prompt + <think>...</think> + JSON-bool; pmass≈1 means model picked a bool token.\n"
        f"=== EVAL demo ({header}) ===\n"
        + _fmt_trace(label, res, tok=tok, schema_hint=frame.get("q"))
        + "\n=== /EVAL ==="
    )


def evaluate_with_vector(model, tok, *, name: str = "scifi", max_think_tokens: int = 64,
                         vector=None, log_demo: bool = True, **kwargs):
    """Run tinymfv.evaluate against `model` with whatever steering is currently
    attached. Pass-through wrapper; emits one decoded demo trace before delegating.

    If `vector` is passed, the demo shows BASE/+C/-C together with pmass+logratio.
    Pass `log_demo=False` for the -C call to avoid duplicating the demo.
    """
    from tinymfv import evaluate
    _log_eval_demo_trace(model, tok, name=name, max_think_tokens=max_think_tokens,
                         vector=vector, log_demo=log_demo)
    return evaluate(model, tok, name=name, max_think_tokens=max_think_tokens, **kwargs)


@torch.no_grad()
def evaluate_multibool(model, tok, *, name: str = "classic", max_think_tokens: int = 256,
                       batch_size: int = 4, foundations: list[str] | None = None) -> dict:
    """Run guided_rollout_multibool on all vignettes × conditions.

    Returns {"raw_logratios": {vid|cond: {f: logratio}}, "raw_pmass": {vid|cond|f: pmass},
    "foundations": [...]}. Logratios are already bias-cancelled (0.5*(lr_viol - lr_ok)).
    Requires a full-attention model (asserts in guided_rollout_multibool).
    """
    from tinymfv.data import load_vignettes
    from tinymfv.core import CONDITIONS
    from tinymfv.guided import guided_rollout_multibool, _DEFAULT_FOUNDATIONS
    from tqdm.auto import tqdm

    if foundations is None:
        foundations = list(_DEFAULT_FOUNDATIONS)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    vignettes = load_vignettes(name)
    raw_logratios: dict[str, dict[str, float]] = {}
    raw_pmass: dict[str, float] = {}

    total = len(vignettes) * len(CONDITIONS)
    with tqdm(total=total, desc="multibool eval", mininterval=60, maxinterval=60) as pbar:
        for cond in CONDITIONS:
            for i in range(0, len(vignettes), batch_size):
                batch = vignettes[i: i + batch_size]
                prompts = [r[cond] for r in batch]
                results = guided_rollout_multibool(
                    model, tok, prompts,
                    foundations=foundations,
                    max_think_tokens=max_think_tokens,
                )
                for r, res in zip(batch, results):
                    key = f"{r['id']}|{cond}"
                    raw_logratios[key] = res.logratios
                    for f in foundations:
                        raw_pmass[f"{key}|{f}"] = res.pmass_format[f]
                pbar.update(len(batch))

    pmass_vals = list(raw_pmass.values())
    pmass_mean = sum(pmass_vals) / len(pmass_vals) if pmass_vals else float("nan")
    n_low = sum(1 for v in pmass_vals if v < 0.5)
    logger.info(
        f"multibool eval: {len(raw_logratios)} (vid,cond) pairs  "
        f"pmass_mean={pmass_mean:.3f}  low_pmass={n_low}/{len(pmass_vals)}"
    )
    # SHOULD: pmass_mean≈1.0 (Qwen3-4B), n_low=0. Else model is hybrid or wrong size.
    return {"raw_logratios": raw_logratios, "raw_pmass": raw_pmass, "foundations": foundations}
