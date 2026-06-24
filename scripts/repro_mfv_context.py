"""Bisect the MFV in-context collapse: isolated tinymfv.evaluate(classic,256) reads
pmass 0.984 (probes 201/205), but inside run_allinstr (job 183) MFV base read
pmass 0.166 / emitted_close 220/264. Same core, same params, same model -> the
cause is process state accumulated before the MFV eval.

This runs the SAME setup as run_allinstr up to (but not including) the steered
evals, evaluating MFV coherence at each stage so we can see WHICH step flips it:

  stage 0  fresh model                          (expect pmass ~0.98, close ~4/264)
  stage 1  after sl.train (mean_diff extract)   (does training forward state flip it?)
  stage 2  after one steered with-block + detach (does a used+detached hook leave residue?)

If stage 1 or 2 reproduces pmass ~0.17 / close ~220, that step is the culprit and
the fix lives in steering-lite, not tinymfv. If all stay clean, the ordinal admin
loop is implicated and we add it here next.

  uv run --extra benchmark python scripts/repro_mfv_context.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering_lite as sl
from steering_lite.data import make_persona_pairs, PERSONA_PAIRS_AUTHORITY
from tinymfv import evaluate


def _mfv(model, tok, tag: str, budget: int, bs: int) -> None:
    rep = evaluate(model, tok, name="classic", max_think_tokens=budget,
                   batch_size=bs, verbose=0)
    logger.info(f"[{tag}] budget={budget}  pmass={rep['mean_pmass_allowed']:.3f}  "
                f"top1={rep['top1_acc']:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--n-pairs", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    logger.info("SHOULD: stage0 pmass ~0.98. A stage where pmass crashes to ~0.17 / "
                "is the culprit for the job-183 MFV collapse.")
    _mfv(model, tok, "stage0-fresh", args.budget, args.batch_size)

    n = model.config.num_hidden_layers
    layers = tuple(range(max(2, int(n * 0.2)), min(n - 2, int(n * 0.8))))
    pos, neg = make_persona_pairs(tok, n_pairs=args.n_pairs, thinking=True,
                                  persona_pairs=PERSONA_PAIRS_AUTHORITY)
    v = sl.train(model, tok, pos, neg,
                 sl.MeanDiffC(layers=layers, coeff=1.0, dtype=torch.bfloat16, seed=0),
                 batch_size=8, max_length=384)
    _mfv(model, tok, "stage1-after-train", args.budget, args.batch_size)

    with v(model, C=1.0):
        pass  # attach + immediate detach, mirroring a steered eval's lifecycle
    _mfv(model, tok, "stage2-after-attach-detach", args.budget, args.batch_size)


if __name__ == "__main__":
    main()
