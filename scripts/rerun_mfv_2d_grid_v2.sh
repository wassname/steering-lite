#!/usr/bin/env zsh
# Re-run ONLY the mfv instrument on the 2D grid (bug fix 5da0fa2).
# The other 3 instruments (big5, humor_styles, mfq2) already completed in pueue 515.
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_2d_grid_showcase.py \
  --model Qwen/Qwen3-4B \
  --vector-a outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8/vector.safetensors \
  --vector-b outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8/vector.safetensors \
  --summary-a outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8/summary.json \
  --summary-b outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8/summary.json \
  --label-a honesty \
  --label-b credulity \
  --hc-grid='-1,-0.5,0,0.5,1' \
  --cc-grid='-1,-0.5,0,0.5,1' \
  --instruments mfv \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --admin-think-tokens 64 \
  --max-think-tokens 256 \
  --out outputs/20260706_honesty_x_credulity_2d_grid_v2_sspace_allinstr_n8
