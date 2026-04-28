#!/usr/bin/env bash
# Run daily_dilemmas benchmark at L=4 with iso-KL-calibrated coeffs.
# Coeffs from outputs/iso_kl/iso_kl__Qwen--Qwen3.5-0.8B__L4__greedy_kl_p95_1.0__T20__N4__seeds0__1777364088.json
set -euo pipefail

MODEL="Qwen/Qwen3.5-0.8B"
LAYERS="4"
OUT="outputs/daily_dilemmas/v5_L4_calibrated"
SEED=0

mkdir -p "$OUT"

run() {
  local method="$1" coeff="$2"
  echo ">>> $method coeff=$coeff"
  uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
    --model "$MODEL" --method "$method" --layers "$LAYERS" \
    --coeff "$coeff" --seed "$SEED" --output-dir "$OUT"
}

run baseline      0.0
run mean_diff     0.9102
run pca           0.8268
run topk_clusters 0.9285
run cosine_gated  6.4400
run sspace        0.8786
run spherical     0.2618

echo "=== all done ==="
ls -la "$OUT"
