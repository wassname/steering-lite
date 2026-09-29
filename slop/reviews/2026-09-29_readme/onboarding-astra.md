## Onboarding verdict

**Partly.** The benchmark pipeline is discoverable, but neither a fresh-account setup nor adding and validating a method is fully documented. Findings below are static observations; execution consequences are inferred, not reproduced.

### Commands I would run

Assuming `uv` and `just` are installed:

```bash
uv sync --extra benchmark --extra test --extra hf-test
uv run --extra benchmark modal setup
# Put OPENROUTER_API_KEY in .env
just smoke-bsbench mean_diff
just sweep dev mean_diff 0 0
just results dev
```

After implementing `my_method.py` following `mean_diff.py`, registering its config/method and exporting its config:

```bash
just smoke-bsbench my_method
just sweep dev my_method 0 0
just results dev
# Inspect dev results before promoting:
just sweep full my_method 0 0
just results full
```

These sweep commands also run **random seed 0**. They are a minimal development sequence, not reproduction of the published three-seed tables.

### Findings

1. **P1 — Setup and cheap-run instructions require inference.** `README.md:13` documents only `uv pip install -e ".[hf-test]"`; benchmark instructions jump to “`just sweep full`” at `README.md:119–123`. I had to inspect `justfile:18–20` to discover method/seed arguments and the additional random sweep, and `pyproject.toml:40–48` to identify benchmark dependencies. Modal authentication and installing `just`/`uv` are undocumented; `modal setup` above is external knowledge. A clean-account walkthrough could disprove the practical blockage, but not the missing instructions.

2. **P1 — A new registered method can pass the advertised check without being tested.** `AGENTS.md:16` promises “every registered method”; actually `tests/test_pipeline.py:30–36,51–78,95` uses a fixed `METHODS` list and config table. `justfile:7,14` additionally runs benchmark smoke only for default `mean_diff`. Thus following the registration instruction at `AGENTS.md:12` does not add coverage. I had to read `mean_diff.py`, `config.py`, package exports and tests to determine the implementation contract and validation gaps. Check by registering a deliberately broken new method and inspecting pytest collection. This also conflicts with “Single functional test = the real benchmark” (`AGENTS.md:11`): `just smoke` runs a separate suite including monkeypatched estimator fixtures (`tests/test_pipeline.py:506`).

3. **P1 — “To reproduce” does not reproduce the displayed experiment.** `README.md:95` says three seeds per method on Qwen; `README.md:122` prescribes plain `just sweep full`. That recipe selects seven named methods and seed `"0"` (`justfile:18`), omitting most table rows and larger models. Existing cached outputs can conceal this mismatch. Validate on an empty output volume, comparing produced certificates against the README table.

4. **P1 — Failed remote walks need not fail the sweep.** `run_modal.main`, `scripts/bsbench/run_modal.py:89–93`, catches every `handle.get()` exception and only prints `"FAILED"`. Consequently `just sweep` can continue into random runs and pulling outputs despite a broken new method. `scripts/bsbench/data.py:25–34` then skips incomplete certificates. Preserve independent-job completion, but report aggregate failure. Disproof: force one remote walk to fail and show a nonzero launcher exit.

5. **P2 — Three smoke entry points have different meanings.** Besides the two just recipes, `scripts/bsbench/run_modal.py:97–100` exposes `::smoke`, running two methods on real 4B GPUs. This is not a dead script, but choosing “smoke” changes method coverage, hardware and cost.

6. **P2 — Calibration prose has drifted inside the implementation, not between README and walk defaults.** README’s RMS/50-token description agrees with `walk.py:275` and `calibrate.py:336,340`. However `_log_kl_history` recommends using “r4ms4e as the target_kl” (`src/steering_lite/calibrate.py:60`), and `:309` calls that statistic the “New default target.” Those instructions disagree with the actual RMS default.

7. **P2 — Results-page build is disconnected.** `results.py:607–619` writes `index.md`, `plot.html` and data, not React `index.html`. Yet `web/uat.py:21` requests `index.html`; `web/package.json:5` only offers bare `vite build`, with no output destination in `web/vite.config.js:4`. Building the explorer requires guessing the output-directory command. No evidence establishes that this actively used page is dead.

8. **P2 — Historical instructions and dense functions slow navigation.** `.pi/goals/01a0d18d-4773-703e-aa3a-7ed67ed54358-e070f2d4.md:134` describes a judge dropdown, superseded only at `:404` by Jev-only cleanup. Mark the old specification archival. In code, `walk()` interleaves diagnostics and the normal benchmark (`walk.py:464–519`); `smooth_path()` compresses its PCHIP slope calculation into one long expression (`results.py:387`). Both require unnecessary reconstruction of control flow/math.