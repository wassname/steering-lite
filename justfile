set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra hf-test --extra benchmark pytest -q tests/test_daily_dilemmas_smoke.py

bench model="Qwen/Qwen3.5-0.8B" method="mean_diff" coeff="2.0":
	uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
		--model {{model}} \
		--method {{method}} \
		--coeff {{coeff}}
