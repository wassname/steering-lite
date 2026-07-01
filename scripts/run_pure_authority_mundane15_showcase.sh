#!/usr/bin/env zsh
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method pca \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair pure_authority \
  --persona-library-template "__verbatim_skill_persona__" \
  --persona-library-scenarios data/persona_library_selections/pure_authority_qwen3_14b_mundane15.jsonl \
  --n-pairs 15 \
  --instruments mfv mfq2 \
  --target-kl 0.5 \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1 \
  --out outputs/20260630_pure_authority_qwen3_14b_mundane15_pca_mfv_mfq2_n8 \
  "$@"
