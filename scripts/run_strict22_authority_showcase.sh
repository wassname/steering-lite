#!/usr/bin/env zsh
set -euo pipefail

uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method sspace \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair dignity_over_authority \
  --persona-library-template "For this answer, use the values of someone {persona} as your own." \
  --persona-library-scenarios data/persona_library_selections/authority_dignity_strict22.jsonl \
  --n-pairs 22 \
  --instruments mfv humor_styles big5 mfq2 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1,2,3,4 \
  --out outputs/20260630_dignity_authority_strict22_local_sspace_mfvgrid_n8
