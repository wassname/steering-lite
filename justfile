set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra hf-test --extra benchmark pytest -q tests/test_smoke.py

bench model="Qwen/Qwen3-0.6B" target="Truthfulness" method="mean_diff" coeff="2.0":
	uv run --extra benchmark python scripts/airisk_dilemmas_benchmark.py \
		--model {{model}} --target {{target}} --method {{method}} --coeff {{coeff}}

# Full README leaderboard sweep: baseline + all methods, auto sign-select.
bench-readme model="Qwen/Qwen3-0.6B" target="Truthfulness" out="outputs/airisk_dilemmas/readme_latest":
	uv run --extra benchmark python scripts/run_airisk_readme_latest.py \
		--model {{model}} --target {{target}} --out {{out}} \
		--device cuda --torch-dtype bfloat16

# Summarize latest airisk outputs into a delta table.
summarize dir target="Truthfulness":
	uv run --extra benchmark python scripts/summarize_airisk_latest.py \
		--dir {{dir}} --target {{target}}
