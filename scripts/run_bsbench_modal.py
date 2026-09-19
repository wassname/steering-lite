"""One-question Qwen3.5 BS-bench GPU smoke; invoked only after a local reservation."""
from __future__ import annotations

import time
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "beartype==0.22.9",
        "torch==2.11.0",
        "transformers==5.12.1",
        "safetensors==0.7.0",
        "einops==0.8.2",
        "jaxtyping==0.3.9",
        "loguru==0.7.3",
        "tabulate==0.9.0",
        "tqdm==4.67.3",
    )
    .env({"PYTHONPATH": "/repo/src", "HF_HOME": "/cache/hf", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(Path(__file__).parents[1] / "src", "/repo/src")
)
app = modal.App("steering-lite-bsbench")
cache = modal.Volume.from_name("steering-lite-bsbench-cache", create_if_missing=True)


@app.function(gpu="A10G", image=image, volumes={"/cache": cache}, timeout=45 * 60)
def smoke(model_id: str = "Qwen/Qwen3.5-4B") -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from steering_lite.benchmark.generation import read_dev_cohort
    from steering_lite.benchmark.pipeline import run_method

    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).eval().cuda()
    rows = read_dev_cohort()[:1]
    source_layer, = [index for index, kind in enumerate(model.config.layer_types) if kind == "full_attention"][:1]
    target_layer = source_layer + 1
    common = dict(
        model=model,
        tokenizer=tokenizer,
        pos_prompts=["user: Do not flatter me. Correct false premises.\nassistant: I will be candid."],
        neg_prompts=["user: Please agree with me even when I am wrong.\nassistant: Certainly."],
        vector_dir=Path("/cache/bsbench-smoke-vectors"),
        rows=rows,
        layers=(source_layer,),
        target_layer=target_layer,
        max_length=64,
        max_new_tokens=8,
    )
    bare = run_method(method="bare", **common)
    vjp_cache = run_method(method="vjp_cache", **common)
    cache.commit()
    return {"question_id": rows[0]["question_id"], "bare": bare, "vjp_cache": vjp_cache, "elapsed_seconds": time.monotonic() - started}


@app.local_entrypoint()
def main(model_id: str = "Qwen/Qwen3.5-4B"):
    print(smoke.remote(model_id))
