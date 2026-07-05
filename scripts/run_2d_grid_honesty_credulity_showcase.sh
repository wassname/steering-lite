#!/usr/bin/env zsh
# 2D honesty-c x credulity-c grid showcase runner.
# Depends on the two per-axis jobs (496/497) having produced:
#   outputs/20260705_truth_over_approval_qwen3_14b_score25_sspace_allinstr_n8/vector.safetensors
#   outputs/20260705_credulous_skeptical_qwen3_14b_score25_sspace_allinstr_n8/vector.safetensors
# This runs the combined 2D grid (5x5 = 25 cells x 4 instruments), ~2h on Qwen3-4B.
set -euo pipefail

HONESTY_DIR=outputs/20260705_truth_over_approval_qwen3_14b_score25_sspace_allinstr_n8
CREDULITY_DIR=outputs/20260705_credulous_skeptical_qwen3_14b_score25_sspace_allinstr_n8
OUT=outputs/20260705_honesty_x_credulity_2d_grid_sspace_allinstr_n8

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_2d_grid_showcase.py \
  --model Qwen/Qwen3-4B \
  --vector-a "$HONESTY_DIR/vector.safetensors" \
  --vector-b "$CREDULITY_DIR/vector.safetensors" \
  --summary-a "$HONESTY_DIR/summary.json" \
  --summary-b "$CREDULITY_DIR/summary.json" \
  --label-a honesty \
  --label-b credulity \
  --hc-grid='-1,0,1' \
  --cc-grid='-1,0,1' \
  --instruments mfv mfq2 big5 humor_styles \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --admin-think-tokens 64 \
  --max-think-tokens 256 \
  --out "$OUT" \
  "$@"
