"""Bench a model preset (config.py) on Modal before its first sweep: peak memory and seconds per dose.

Same prompts and generation code as walk.py; times one dose of 200 answers (the 100 BS-bench questions at +C and
-C) after a warm-up batch. bare = unsteered (healthy, short answers); broken (4B only) = a mean_diff vector past
breakdown, so every answer runs to the token cap, the worst case for padding waste. Record the result in the
preset's `measured` field (AGENTS.md). History: journal 2026-10-04 (GPU x batch table; causal-conv1d kernel ~1-5%).

  uv run --extra benchmark modal run scripts/bsbench/bench_modal.py::main --preset qwen3.5-4b
  uv run --extra benchmark modal run scripts/bsbench/bench_modal.py::main --preset qwen3.5-4b --gpus L4,A10G --batches 128,200
"""
import os
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
sys.path.insert(0, str(REPO / "scripts/bsbench"))
from config import PRESETS  # noqa: E402

RATE = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}  # $/h, modal.com/pricing 2026-10-04
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install("torch==2.11.0", "transformers==5.12.1", "accelerate==1.13.0", "safetensors==0.7.0", "einops==0.8.2", "jaxtyping==0.3.9",
                    "beartype==0.22.9", "loguru==0.7.3", "tabulate==0.10.0", "tqdm==4.67.3", "numpy==2.4.4", "flash-linear-attention==0.5.2",
                    "fla-core==0.5.2", "tyro==1.0.16")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "PYTHONPATH": "/repo/src:/repo/scripts/bsbench"})
    .add_local_dir(REPO / "src", "/repo/src").add_local_dir(REPO / "scripts", "/repo/scripts").add_local_dir(REPO / "data", "/repo/data")
)
app = modal.App("steering-lite-bsbench-bench", image=image)
cache = modal.Volume.from_name("steering-lite-bsbench-v3")
VECTOR = "/cache/bsbench/Qwen--Qwen3.5-4B-g7e7c6071/vectors/mean_diff_s0.safetensors"  # a 4B mean_diff vector for the broken case


@app.function(volumes={"/cache": cache}, timeout=3600)
def bench(preset: str, batches: list[int], broken_C: float) -> list[dict]:
    os.chdir("/repo")
    import torch
    import walk
    from steering_lite import Vector
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = PRESETS[preset]
    cap = walk.GEN["max_new_tokens"]
    tokenizer = AutoTokenizer.from_pretrained(p.model)
    model = AutoModelForCausalLM.from_pretrained(p.model, dtype=getattr(torch, p.dtype), device_map="cuda").eval()
    prompts = walk.generation_inputs(tokenizer, walk.read_cohort("full")) * 2  # one dose: 100 questions at +C and -C
    conditions = [("bare", 0.0)] + ([("broken", broken_C)] if p.model == "Qwen/Qwen3.5-4B" else [])
    vector = Vector.load(VECTOR) if len(conditions) == 2 else None
    out = []
    for condition, C in conditions:
        for batch in batches:
            steer = (lambda: vector(model, C=-C)) if C else walk._Null
            with steer():
                walk.generate(model, tokenizer, prompts[:batch], batch)  # warm-up
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                answers = walk.generate(model, tokenizer, prompts, batch)
                torch.cuda.synchronize()
            tokens = [len(tokenizer(a, add_special_tokens=False)["input_ids"]) for a in answers]
            out.append({"condition": condition, "batch": batch, "seconds": time.monotonic() - start, "answers": len(answers),
                        "mean_tokens": sum(tokens) / len(tokens), "max_tokens": max(tokens), "at_cap": sum(t >= cap - 1 for t in tokens),
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "total_gb": torch.cuda.get_device_properties(0).total_memory / 1e9})
            print(out[-1], flush=True)
    return out


@app.local_entrypoint()
def main(preset: str = "qwen3.5-4b", gpus: str = "", batches: str = "", broken_c: float = 1.26):
    p = PRESETS[preset]
    sizes = [int(b) for b in batches.split(",")] if batches else [p.batch_size]
    calls = {gpu: bench.with_options(gpu=gpu).spawn(preset, sizes, broken_c) for gpu in (gpus.split(",") if gpus else [p.gpu])}
    for gpu, call in calls.items():
        for r in call.get():
            per_1k = RATE[gpu] * r["seconds"] / 3600 / r["answers"] * 1000
            print(f"BENCH preset={preset} gpu={gpu} {r['condition']:6} batch={r['batch']:3} s={r['seconds']:6.1f} $/1k answers={per_1k:.3f} "
                  f"mean_tok={r['mean_tokens']:.0f} max_tok={r['max_tokens']} at_cap={r['at_cap']} peak_gb={r['peak_gb']:.1f} of {r['total_gb']:.0f}")
