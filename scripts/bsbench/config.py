"""Model presets: one per model size, holding what to load and how to fill the GPU it runs on.

Intent (wassname 2026-10-04): every model gets its own preset, and a preset is benched with
`scripts/bsbench/bench_modal.py --preset NAME` (one 200-answer dose, peak memory and seconds) before its
first sweep; record the measurement here. A small model on a big GPU at a small batch wastes most of the
money: decoding is per-step overhead bound, so fill the batch first (journal 2026-10-04).
"""
from dataclasses import dataclass

import tyro


@dataclass(frozen=True)
class ModelPreset:
    model: str
    gpu: str  # Modal GPU type
    batch_size: int  # generation batch; aim to fill the GPU, keep peak memory under ~85% of it
    dtype: str = "bfloat16"
    device: str = "cuda"
    measured: str = ""  # bench_modal.py result: peak GB, seconds per 200-answer dose (healthy / broken), date


PRESETS = {
    "qwen3.5-4b": ModelPreset(
        model="Qwen/Qwen3.5-4B", gpu="A10G", batch_size=200,
        measured="2026-10-04 A10G batch 200: peak 18.3 GB of 24; 20.4 s healthy / 34.7 s broken per 200 answers",
    ),
    # TODO test and check peak mem: bf16 weights are about 54 GB; bench before the first sweep.
    "qwen3.5-27b": ModelPreset(model="Qwen/Qwen3.5-27B", gpu="H100", batch_size=64),
    # CPU smoke tests (`just smoke-bsbench`): tiny random model, not benched.
    "tiny-random": ModelPreset(model="wassname/qwen3-5lyr-tiny-random", gpu="", batch_size=8, dtype="float32", device="cpu"),
}


def preset_cli(args: list[str] | None = None) -> ModelPreset:
    """`--preset NAME` plus field overrides, e.g. `qwen3.5-4b --batch-size 128`."""
    return tyro.extras.overridable_config_cli({name: (name, preset) for name, preset in PRESETS.items()}, args=args)
