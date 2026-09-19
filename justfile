set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra hf-test --extra benchmark pytest -q tests/test_pipeline.py tests/test_benchmark_pipeline.py

# Non-paid preflight: writes the Qwen3.5-4B manifest without network, model, judge, or Modal calls.
sweep mode="--dry-run" model="Qwen/Qwen3.5-4B" out="outputs/bsbench-v2":
	.venv/bin/python scripts/run_bsbench_sweep.py {{mode}} --model {{model}} --out {{out}}
