"""Pin the MFV collapse to the adapter's demo trace.

run_allinstr's MFV eval (evaluate_multibool, WITH _log_eval_demo_trace) reads
pmass 0.166 deterministically (jobs 183, 210 identical to 4 digits). The direct
tinymfv.evaluate (NO demo) reads pmass 0.984 (probes 201/205/206/208). The only
difference is the demo trace: a bs=1 guided_rollout_forced_choice that itself
returns NaN in job 183/210. Hypothesis: the bs=1 NaN forward leaves persistent
corruption (Qwen3.5 gated-delta-net recurrent state) that poisons the next eval.

Two calls on the SAME fresh model, demo off then on:

  A  evaluate_multibool(log_demo=False)   -> expect pmass ~0.98 (no demo)
  B  evaluate_multibool(log_demo=True)    -> expect pmass ~0.17 IF the demo poisons

If B << A, the adapter demo trace is the culprit; fix lives in steering-lite
(drop/repair the demo) not tinymfv.

  uv run --extra benchmark python scripts/repro_mfv_demo_poison.py --model Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from steering_lite.eval.tinymfv import evaluate_multibool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to(args.device).eval()

    logger.info("SHOULD: A (demo off) pmass ~0.98; B (demo on) ~0.98 too UNLESS the "
                "bs=1 demo trace poisons the model -> B ~0.17.")
    for tag, log_demo in [("A-demo-off", False), ("B-demo-on", True)]:
        rep = evaluate_multibool(model, tok, name="classic", max_think_tokens=args.budget,
                                 batch_size=args.batch_size, log_demo=log_demo)
        # evaluate_multibool returns top1_acc; pmass lives in info / per_row margins.
        # mean coherence: recompute from per_row pmass is not exposed, so read top1_acc
        # plus the underlying tinymfv mean_pmass via info.
        pm = rep["info"].get("mean_pmass_allowed", float("nan"))
        logger.info(f"[{tag}] pmass={pm:.3f}  top1={rep['top1_acc']:.3f}")


if __name__ == "__main__":
    main()
