#!/usr/bin/env zsh
# v2 Honesty axis (truth_over_approval) showcase runner.
# Template + scenarios from v2 Stage A/B validation with stratified sampling + on-axis dim exclusion.
# Winner template: "Use the priorities of a {persona} person." (strict 8.3%, axis_delta +2.03).
# Scenarios: strict-pass only at axis_delta>=2.0, off-axis<=2.0, excluding on-axis confound dims
# (honesty_truthfulness, praise_flattery, sycophancy). 24 strict-pass pairs.
# c-grid "0.5,1" yields the symmetric signed sweep {-1,-0.5,0,+0.5,+1}.
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method sspace \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair truth_over_approval \
  --persona-library-template "Use the priorities of a {persona} person." \
  --persona-library-scenarios data/persona_library_selections/truth_over_approval_strict_v2.jsonl \
  --n-pairs 24 \
  --instruments mfv mfq2 big5 humor_styles \
  --target-kl 0.5 \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1 \
  --out outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8 \
  --save-vector outputs/20260706_truth_over_approval_strict_v2_sspace_allinstr_n8/vector.safetensors \
  "$@"
