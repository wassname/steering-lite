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
