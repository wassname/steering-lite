#!/usr/bin/env bash
# v8: Bench at L=7..18 (mid 30-80% of 24 layers) with iso-KL coeffs from
# guided-thinking probes. Uses guided CoT eval (32 think tokens under
# steering, then '</think>\nMy choice:' force-close, score Yes/No).
# Coeffs from outputs/iso_kl/iso_kl__Qwen--Qwen3.5-0.8B__L7_8_9_10_11_12_13_14_15_16_17_18__greedy_kl_p95_1.0__T20__N4__seeds0__1777395640.json
set -euo pipefail

MODEL="Qwen/Qwen3.5-0.8B"
LAYERS="7,8,9,10,11,12,13,14,15,16,17,18"
OUT="outputs/daily_dilemmas/v8_guided"
SEED=0

mkdir -p "$OUT"

run() {
  local method="$1" coeff="$2"
  echo ">>> $method coeff=$coeff"
  uv run --extra benchmark python scripts/daily_dilemmas_benchmark.py \
    --model "$MODEL" --method "$method" --layers "$LAYERS" \
    --coeff "$coeff" --seed "$SEED" --output-dir "$OUT" \
    --device cuda --torch-dtype bfloat16 \
    --max-think-tokens 32
}

run baseline      0.0
run mean_diff     0.1486
run pca           0.2841
run topk_clusters 0.2386
run cosine_gated  1.9682
run sspace        0.1481
run spherical     0.0502

echo "=== all done ==="
ls -la "$OUT"
