set shell := ["bash", "-cu"]

default:
	@just --list

check: test smoke

test:
	uv run --extra test --extra hf-test --extra benchmark pytest -q

smoke:
	uv run --extra test --extra hf-test --extra benchmark pytest -q tests/test_daily_dilemmas_smoke.py

bench model="Qwen/Qwen3-0.6B" method="mean_diff" coeff="2.0":
	uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
		--model {{model}} --method {{method}} --coeff {{coeff}}

# Bidirectional weight-steering bench: each method at +c, -c -> SI via si-flip.
bench-bidir model="Qwen/Qwen3-0.6B" out="outputs/daily_dilemmas/bidir_qwen" layers="mid" coeff="1.0":
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p {{out}}
	uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
		--model {{model}} --method baseline --coeff 0.0 \
		--layers {{layers}} --device cuda --torch-dtype bfloat16 --output-dir {{out}}
	for method in mean_diff pca topk_clusters cosine_gated sspace spherical; do
		for sign in "+" "-"; do
			uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
				--model {{model}} --method $method --coeff "${sign}{{coeff}}" \
				--layers {{layers}} --device cuda --torch-dtype bfloat16 --output-dir {{out}}
		done
	done

# Bidir bench at iso-KL-calibrated coeffs (per-method c*).
bench-iso iso_kl out:
	uv run --extra benchmark python scripts/bench_bidir_iso.py \
		--iso-kl {{iso_kl}} --out {{out}}

# Aggregate per_row.csv into SI table.
si-flip dir:
	uv run --extra benchmark python scripts/compute_si_flip.py {{dir}}
