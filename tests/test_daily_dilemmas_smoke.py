"""Single end-to-end functional test: run the daily-dilemmas benchmark on
synthetic pairs (offline) for every method + the baseline. Same code path as
the real bench.

Asserts per method:
  - extracted steering state is non-zero
  - steering produces a measurable per-value delta
  - save/load round-trip preserves logits
For 'baseline' (no steering): SI must be exactly 0.
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


METHODS = ["mean_diff", "caa", "act_add", "mean_centred", "pca", "topk_clusters", "cosine_gated", "sspace", "spherical", "directional_ablation", "chars", "linear_act", "angular_steering", "baseline"]
TINY_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def quick_cfg(method: str, tmp_path: Path) -> "benchmark.BenchmarkConfig":
    return benchmark.BenchmarkConfig(
        model=TINY_MODEL,
        method=method,
        target="honesty",
        layers=(1,),
        coeff=0.1 if method in ("spherical", "angular_steering", "linear_act") else 2.0,
        n_train=8,
        n_eval=4,
        max_seq_length=64,
        torch_dtype="float32",
        device="cpu",
        seed=0,
        synthetic=True,
        output_dir=tmp_path / "out",
        save_load_check=(method != "baseline"),
    )


@pytest.mark.parametrize("method", METHODS)
def test_full_pipeline(method: str, tmp_path: Path):
    cfg = quick_cfg(method, tmp_path)
    result = benchmark.run(cfg)

    summary = result["summary"]
    assert math.isfinite(summary["target_effect"]), (
        f"{method}: non-finite target_effect={summary['target_effect']}"
    )
    # leakage / SI may be NaN in synthetic mode (no other COMMON_VALUES present)

    if method == "baseline":
        assert summary["target_effect"] == 0.0, "baseline must not shift target"
        return

    norms = result["vector_norms"]
    assert 1 in norms, f"{method}: missing layer 1 in vectors"
    layer_norms = norms[1]
    assert layer_norms and any(v > 0 for v in layer_norms.values()), (
        f"{method}: all state tensors zero -- extract silently broken: {layer_norms}"
    )

    deltas = [e["delta"] for e in result["effects"].values() if e["n"] > 0]
    assert any(abs(d) > 1e-6 for d in deltas), (
        f"{method}: zero delta on every value -- hook didn't fire? deltas={deltas}"
    )

    err = result["save_load_err"]
    assert err is not None and err < cfg.save_load_tol, (
        f"{method}: save/load mismatch err={err} > tol={cfg.save_load_tol}"
    )


def test_not_to_do_rows_are_not_sign_flipped(monkeypatch):
    def fake_score_mcq(*args, **kwargs):
        return {"logratio": 2.0, "pmass": 0.5, "max_p": 0.6, "logp": None, "think_tokens": 0}

    monkeypatch.setattr(benchmark, "score_mcq", fake_score_mcq)
    # value_label encodes the sign: +1 = Yes-expresses-target, -1 = Yes-opposes-target.
    # logratio_act = logratio * value_label, so a row where the target answer
    # is No (label=-1) flips sign correctly even when logratio>0 (model says Yes).
    rows = [
        benchmark.DilemmaRow(0, "to_do", "s", "Continue stealing", value_label=-1.0),
        benchmark.DilemmaRow(0, "not_to_do", "s", "Stop stealing", value_label=+1.0),
    ]

    scores, per_row = benchmark._eval_run(None, None, rows, choice_ids=[[], []], device="cpu")

    assert per_row[0]["logratio_act"] == -2.0  # Yes endorses dishonest -> negative
    assert per_row[1]["logratio_act"] == +2.0  # Yes endorses honest -> positive
    assert scores == [-2.0, +2.0]
