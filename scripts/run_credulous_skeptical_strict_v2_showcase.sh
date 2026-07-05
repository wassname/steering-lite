#!/usr/bin/env zsh
# v2 Credulity axis (credulous_skeptical) showcase runner.
# Template + scenarios from v2 Stage A/B validation with stratified sampling.
# Winner template: "For this answer, use the values of someone {persona} as your own." (strict 16.7%, ax_d +2.37).
# Scenarios: strict-pass only at axis_delta>=2.0, off-axis<=2.0.
# No on-axis dim exclusion needed for credulity (none of the confound dims are circularly on-axis:
# credulity is about epistemic trust, not truth-telling/sycophancy/flattery which are behavioral style).
# c-grid "0.5,1" yields the symmetric signed sweep {-1,-0.5,0,+0.5,+1}.
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method sspace \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair credulous_skeptical \
  --persona-library-template "For this answer, use the values of someone {persona} as your own." \
  --persona-library-scenarios data/persona_library_selections/credulous_skeptical_strict_v2.jsonl \
  --n-pairs 50 \
  --instruments mfv mfq2 big5 humor_styles \
  --target-kl 0.5 \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1 \
  --out outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8 \
  --save-vector outputs/20260706_credulous_skeptical_strict_v2_sspace_allinstr_n8/vector.safetensors \
  "$@"
