#!/usr/bin/env bash
# Bench daily_dilemmas at L=7..19 (30-80% of 24 layers) with iso-KL-calibrated coeffs.
# Coeffs from outputs/iso_kl/iso_kl__Qwen--Qwen3.5-0.8B__L7_8_9_10_11_12_13_14_15_16_17_18_19__greedy_kl_p95_1.0__T20__N4__seeds0__1777371163.json
set -euo pipefail

MODEL="Qwen/Qwen3.5-0.8B"
LAYERS="7,8,9,10,11,12,13,14,15,16,17,18,19"
OUT="outputs/daily_dilemmas/v6_L7-19_calibrated"
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
run mean_diff     0.1659
run pca           0.7417
run topk_clusters 0.2272
run cosine_gated  1.9485
run sspace        0.1474
run spherical     0.0379

echo "=== all done ==="
ls -la "$OUT"
