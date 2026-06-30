#!/usr/bin/env zsh
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method sspace \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair authority_tradition_obedience \
  --persona-library-template "Answer as someone who is {persona}." \
  --persona-library-scenarios data/persona_library_selections/authority_only_qwen3_14b_score60.jsonl \
  --n-pairs 32 \
  --instruments mfv mfq2 \
  --target-kl 0.5 \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1,2,3,4 \
  --out outputs/20260630_authority_only_qwen3_14b_score60_sspace_mfv_mfq2_n8 \
  "$@"
