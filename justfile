set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra hf-test --extra benchmark pytest -q tests/test_pipeline.py

# Full sweep: extract -> calibrate -> tinymfv guided CoT eval, one row per method.
sweep model="Qwen/Qwen3-0.6B" out="outputs/tinymfv_sweep":
	uv run --extra benchmark python scripts/run_tinymfv_sweep.py \
		--model {{model}} --out {{out}} \
		--device cuda --torch-dtype bfloat16

# Render README-ready tables (base vs humans, SI, Δlogit ± σ, sign agreement)
# and a moral-map PNG (PCA scatter + ΔAuth-vs-ΔCare surgical view).
results sweep_dir="outputs/tinymfv_sweep" vignettes="airisk":
	uv run --extra benchmark --with matplotlib --with adjustText python scripts/results.py \
		--sweep-dir {{sweep_dir}} --vignettes {{vignettes}}
