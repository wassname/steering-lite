# AGENTS.md

Inherits conventions from sibling project `lora-lite`. Read [../lora-lite/AGENTS.md](../lora-lite/AGENTS.md) if it exists.

## House rules

- Fail fast. No defensive programming, no fallbacks, no silent dequant.
- One file per method under `src/steering_lite/variants/`. Docstring -> paper link, math, intuition.
- Use `einops.einsum` and `jaxtyping` shape annotations. Tensor dim letters: `b s d` (batch, seq, d_model), `n` (prompts), `r` (rank/components), `k` (clusters), `l` (layer).
- No backward compat. Break things to gain simplicity.
- Keep tests to smoke tests of the real pipeline at tiny scale (`just check`). Don't add a separate unit-test suite.
- New methods register via `@register_config` and `@register` decorators; export `Config` class from `__init__.py`; add the name to `METHODS` in `tests/test_pipeline.py` (a test fails until you do).

## Verify

`just smoke` -> every registered method passes extract -> attach -> generate -> save/load on tiny-random-Llama. It asserts non-zero state, a non-zero residual delta, and a save/load round-trip below `1e-4`.

`just dev METHOD` -> the method on Qwen3.5-9B, seed 0, 20 questions, compared with every finished full walk sliced to the same questions (`outputs/bsbench/results/v5-9b-dev`). Costs GPU and judge credits.

`just smoke-bsbench` -> the BS-bench dose walk (`scripts/bsbench/walk.py`) runs 2 rungs on a tiny random Qwen3 on CPU. `just check` runs both.

## Benchmark

`scripts/bsbench/`: `walk.py` (dose walk) -> `judge.py` (Jev ratings) -> `results.py` (tables, plot, `points.json`). Results and their evidence go in `RESEARCH_JOURNAL.md`.

Models run from presets in `scripts/bsbench/config.py` (model, dtype, Modal GPU, batch size). Before the first sweep on a new or changed preset, run `uv run --extra benchmark modal run scripts/bsbench/bench_modal.py::main --preset NAME` and record peak GB and seconds per dose in the preset's `measured` field. Fill the batch first: a 4B model at batch 32 on an L40S cost 3-5x more per answer than batch 200 on an A10G (journal 2026-10-04). A preset with an empty `measured` field has not been benched.

Coherence for plots and scores comes only from Jev (mean damage ≤1.5/4). Mechanical checks are calibration diagnostics, not plot/scoring exclusions; a mechanical boundary does not make a judged point inadmissible. — PI/OpenAI, wassname clarification 2026-10-02.
