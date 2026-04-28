"""Single end-to-end functional test: run the daily-dilemmas benchmark for every
method on tiny-random Llama. Same code path as the real bench, just smaller args.

Asserts per method:
  - extracted steering state is non-zero
  - steered generation differs from baseline (save/load round-trip < tol means
    state was actually installed)
  - target effect is well-defined (finite)

We can't assert `target_effect > 0` strictly on tiny-random because the model
has random weights and probe tokens have no semantic grounding -- but we can
assert the steering changes outputs. The full bench on a real model is where
surgical informedness is meaningful.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import math
import pytest

SPEC = importlib.util.spec_from_file_location(
    "daily_dilemmas_benchmark",
    Path(__file__).resolve().parent.parent / "scripts" / "daily_dilemmas_benchmark.py",
)
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


METHODS = ["mean_diff", "pca", "topk_clusters", "cosine_gated", "sspace", "spherical"]
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def quick_cfg(method: str, tmp_path: Path) -> "benchmark.BenchmarkConfig":
    cfg = benchmark.BenchmarkConfig(
        model=TINY_MODEL,
        method=method,
        layers=(1,),
        coeff=0.1 if method == "spherical" else 2.0,  # spherical t in [0, 1]
        n_train=8,
        n_eval=4,
        max_new_tokens=8,
        max_seq_length=32,
        torch_dtype="float32",
        device="cpu",
        seed=0,
        output_dir=tmp_path / "out",
        save_load_check=True,
    )
    return cfg


@pytest.mark.parametrize("method", METHODS)
def test_full_pipeline(method: str, tmp_path: Path):
    cfg = quick_cfg(method, tmp_path)
    result = benchmark.run(cfg)

    # 1. extract produced non-zero state at the selected layer
    norms = result["vector_norms"]
    assert 1 in norms, f"missing layer 1 in vectors: {norms}"
    layer_norms = norms[1]
    assert layer_norms, f"no state tensors stored for {method}"
    assert any(v > 0 for v in layer_norms.values()), (
        f"{method}: all state tensors zero -- extract is silently broken: {layer_norms}"
    )

    # 2. base vs steered probe-token scores are finite (model ran end-to-end)
    for k in ("target", "control"):
        assert math.isfinite(result["base"][k]), f"{method}: non-finite base[{k}]"
        assert math.isfinite(result["steered"][k]), f"{method}: non-finite steered[{k}]"

    # 3. steering changed output -- target effect or control effect must be nonzero,
    #    else the hook silently did nothing.
    eff_t = result["effect"]["target"]
    eff_c = result["effect"]["control"]
    assert abs(eff_t) > 1e-6 or abs(eff_c) > 1e-6, (
        f"{method}: zero effect on both target and control -- hook didn't fire? "
        f"effect={result['effect']}"
    )

    # 4. save/load round-trip preserves steered logits exactly (within tol)
    err = result["save_load_err"]
    assert err is not None and err < cfg.save_load_tol, (
        f"{method}: save/load mismatch err={err} > tol={cfg.save_load_tol}"
    )
