#!/usr/bin/env zsh
# v2 2D honesty x credulity grid showcase runner.
# Depends on the two per-axis v2 jobs having produced:
#   outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8/vector.safetensors
#   outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8/vector.safetensors
# 5x5 grid (25 cells x 4 instruments), pub-quality resolution.
set -euo pipefail

HONESTY_DIR=outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8
CREDULITY_DIR=outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8
OUT=outputs/20260706_honesty_x_credulity_2d_grid_v2_sspace_allinstr_n8

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_2d_grid_showcase.py \
  --model Qwen/Qwen3-4B \
  --vector-a "$HONESTY_DIR/vector.safetensors" \
  --vector-b "$CREDULITY_DIR/vector.safetensors" \
  --summary-a "$HONESTY_DIR/summary.json" \
  --summary-b "$CREDULITY_DIR/summary.json" \
  --label-a honesty \
  --label-b credulity \
  --hc-grid='-1,-0.5,0,0.5,1' \
  --cc-grid='-1,-0.5,0,0.5,1' \
  --instruments mfv mfq2 big5 humor_styles \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --admin-think-tokens 64 \
  --max-think-tokens 256 \
  --out "$OUT" \
  "$@"
