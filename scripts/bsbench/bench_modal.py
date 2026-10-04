"""Generation throughput on Modal GPUs for the BS-bench walk workload: which GPU x batch is cheapest per answer.

Same prompts, model and generation code as walk.py; times one dose of 200 prompts (100 bench + 100 twins)
after a warm-up batch. bare = unsteered (healthy, short answers); broken = mean_diff at a dose past breakdown
(answers run to the token cap), the worst case for padding waste.

  uv run --extra benchmark modal run scripts/bsbench/bench_modal.py::main --gpus L4,A10,L40S,A100-40GB --batches 32,64,128
  uv run --extra benchmark modal run scripts/bsbench/bench_modal.py::main --conv   # image with causal-conv1d (torch 2.10)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/repo")
RATE = {"T4": 0.59, "L4": 0.80, "A10": 1.10, "A10G": 1.10, "L40S": 1.95, "A100-40GB": 2.10, "A100-80GB": 2.50, "H100": 3.95}  # $/h, modal.com/pricing 2026-10-04
PACKAGES = ("transformers==5.12.1", "accelerate==1.13.0", "safetensors==0.7.0", "einops==0.8.2", "jaxtyping==0.3.9", "beartype==0.22.9",
            "loguru==0.7.3", "tabulate==0.10.0", "tqdm==4.67.3", "numpy==2.4.4", "flash-linear-attention==0.5.2", "fla-core==0.5.2")
CONV_WHEEL = "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu13torch2.10cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"


def _image(conv: bool) -> modal.Image:
    image = modal.Image.debian_slim(python_version="3.13")
    if conv:  # the causal-conv1d wheel is built for torch 2.10 + CUDA 13; PyPI's torch 2.10 is CUDA 12
        image = image.uv_pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130").uv_pip_install(CONV_WHEEL, *PACKAGES)
    else:
        image = image.uv_pip_install("torch==2.11.0", *PACKAGES)
    return (image
            .env({"PYTHONUNBUFFERED": "1", "HF_HOME": "/cache/hf", "PYTHONPATH": "/repo/src:/repo/scripts/bsbench"})
            .add_local_dir(REPO / "src", "/repo/src").add_local_dir(REPO / "scripts", "/repo/scripts").add_local_dir(REPO / "data", "/repo/data"))


app = modal.App("steering-lite-bsbench-bench")
cache = modal.Volume.from_name("steering-lite-bsbench-v3")
VECTOR = "/cache/bsbench/Qwen--Qwen3.5-4B-g7e7c6071/vectors/mean_diff_s0.safetensors"


def _bench(batches: list[int], max_new_tokens: int, broken_C: float) -> list[dict]:
    os.chdir("/repo")
    import torch
    import walk
    from steering_lite import Vector
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qm

    walk.GEN["max_new_tokens"] = max_new_tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", dtype=torch.bfloat16, device_map="cuda").eval()
    rows = walk.read_cohort("full") + walk.read_twins("full")
    prompts = walk.generation_inputs(tokenizer, rows)
    vector = Vector.load(VECTOR)
    out = []
    for condition, C in (("bare", 0.0), ("broken", broken_C)):
        for batch in batches:
            steer = (lambda: vector(model, C=-C)) if C else walk._Null
            with steer():
                walk.generate(model, tokenizer, prompts[:batch], batch)  # warm-up
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                answers = walk.generate(model, tokenizer, prompts, batch)
                torch.cuda.synchronize()
            seconds = time.monotonic() - start
            tokens = [len(tokenizer(a, add_special_tokens=False)["input_ids"]) for a in answers]
            out.append({"condition": condition, "batch": batch, "seconds": seconds, "answers": len(answers),
                        "mean_tokens": sum(tokens) / len(tokens), "max_tokens": max(tokens), "at_cap": sum(t >= max_new_tokens - 1 for t in tokens),
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9, "gpu": torch.cuda.get_device_name(),
                        "conv_kernel": qm.causal_conv1d_fn is not None, "torch": torch.__version__})
            print(out[-1], flush=True)
    return out


@app.function(image=_image(False), volumes={"/cache": cache}, timeout=3600)
def bench(batches: list[int], max_new_tokens: int, broken_C: float) -> list[dict]:
    return _bench(batches, max_new_tokens, broken_C)


@app.function(image=_image(True), volumes={"/cache": cache}, timeout=3600)
def bench_conv(batches: list[int], max_new_tokens: int, broken_C: float) -> list[dict]:
    return _bench(batches, max_new_tokens, broken_C)


@app.local_entrypoint()
def main(gpus: str = "L40S", batches: str = "32,64,128", max_new_tokens: int = 192, broken_c: float = 1.26, conv: bool = False):
    fn = bench_conv if conv else bench
    sizes = [int(b) for b in batches.split(",")]
    calls = {gpu: fn.with_options(gpu=gpu).spawn(sizes, max_new_tokens, broken_c) for gpu in gpus.split(",")}
    for gpu, call in calls.items():
        for r in call.get():
            per_1k = RATE[gpu] * r["seconds"] / 3600 / r["answers"] * 1000
            print(f"BENCH gpu={gpu} conv={r['conv_kernel']} torch={r['torch']} {r['condition']:6} batch={r['batch']:3} "
                  f"s={r['seconds']:6.1f} $/1k answers={per_1k:.3f} mean_tok={r['mean_tokens']:.0f} max_tok={r['max_tokens']} "
                  f"at_cap={r['at_cap']} peak_gb={r['peak_gb']:.1f}")
