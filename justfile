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

# v9: prompt baselines (system-prompt control). One coeff=0 invocation per
# preset, method=baseline (no weight steering). Uses whole dataset.
# n_eval omitted -> all party='You' rows (~2k).
bench-prompt model="Qwen/Qwen3-0.6B" out="outputs/daily_dilemmas/v9_prompt_qwen":
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p {{out}}
	for preset in base simple_honest simple_dishonest engineered_honest engineered_dishonest; do
		echo ">>> prompt preset=$preset"
		uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
			--model {{model}} --method baseline --coeff 0.0 \
			--prompt-preset $preset \
			--device cuda --torch-dtype bfloat16 \
			--output-dir {{out}}
	done

# v9: bidirectional weight-steering bench. Runs each method at +c, -c so
# compute_si_flip.py can compute the canonical bidirectional SI.
# `coeffs` is a comma-separated list incl. 0.0 (only used for `baseline`).
bench-bidir model="Qwen/Qwen3.5-0.8B" out="outputs/daily_dilemmas/v9_bidir_qwen" layers="mid":
	#!/usr/bin/env bash
	set -euo pipefail
	mkdir -p {{out}}
	# baseline ref at coeff=0
	uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
		--model {{model}} --method baseline --coeff 0.0 \
		--layers {{layers}} --device cuda --torch-dtype bfloat16 \
		--output-dir {{out}}
	# active methods at +c, -c (calibrated coeffs from iso-KL probe)
	# fallback default coeff=1.0; replace with iso-KL coeffs once recalibrated.
	for method in mean_diff pca topk_clusters cosine_gated sspace spherical; do
		for sign in +1 -1; do
			coeff=$(python -c "print(${sign} * 1.0)")
			echo ">>> $method coeff=$coeff"
			uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
				--model {{model}} --method $method --coeff $coeff \
				--layers {{layers}} --device cuda --torch-dtype bfloat16 \
				--output-dir {{out}}
		done
	done

# Aggregate per_row.csv files into bidirectional flip-based SI table.
si-flip dir="outputs/daily_dilemmas/v9_bidir_qwen":
	uv run --extra benchmark python scripts/compute_si_flip.py {{dir}}

# Queue prompt + bidir runs for both Qwen3-0.6B and gemma-3-4b-it via pueue.
# Why: get a complete v9 table (prompt baseline + dW bidir for both models).
# Resolve: pueue log will show summary tables per task; aggregate w/ just si-flip.
pueue-v9:
	#!/usr/bin/env bash
	set -euo pipefail
	for model in "Qwen/Qwen3-0.6B" "google/gemma-3-4b-it"; do
		tag=$(echo $model | tr '/' '_')
		pueue add -o 5 -l "v9-prompt $model | why: prompt baseline missing | resolve: si-flip table" \
			-w "$PWD" -- just bench-prompt "$model" "outputs/daily_dilemmas/v9_prompt_${tag}"
		pueue add -o 5 -l "v9-bidir $model | why: bidir SI needs +/-c | resolve: si-flip table" \
			-w "$PWD" -- just bench-bidir "$model" "outputs/daily_dilemmas/v9_bidir_${tag}"
	done
	pueue status
