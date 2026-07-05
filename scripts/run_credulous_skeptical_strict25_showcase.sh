#!/usr/bin/env zsh
# Credulity axis (credulous_skeptical) showcase runner.
# Mirrors run_pure_authority_strict25_showcase.sh. Template + scenario selection are finalized
# by the persona-template-library Stage A/B validation (docs/spec/20260630_authority_steer_pipeline.md
# analogue); the scenario file below is produced by goal 2 of .pi/plan.md and must exist before
# this runner is queued via pueue. c-grid "0.5,1" yields the symmetric signed sweep {-1,-0.5,0,+0.5,+1}.
# Scenario-selection caveat (oracle confound review): avoid scenarios where the only observable is
# "agreement with a user's claim", since the sycophant pole of truth_over_approval (A-neg) and the
# credulous pole here (B-pos) both produce agreement; prefer scenarios that probe belief-formation
# vs output-truthfulness separately.
set -euo pipefail

PYTHONUNBUFFERED=1 uv run --extra benchmark python scripts/run_allinstr_showcase.py \
  --model Qwen/Qwen3-4B \
  --method sspace \
  --pairs-source persona_library \
  --persona-library-dir /media/wassname/SGIronWolf/projects5/2026/weight-steering-repos/persona-steering-template-library \
  --persona-library-pair credulous_skeptical \
  --persona-library-template "Speak with the priorities of someone {persona}." \
  --persona-library-scenarios data/persona_library_selections/credulous_skeptical_qwen3_14b_strict25.jsonl \
  --n-pairs 25 \
  --instruments mfv mfq2 big5 humor_styles \
  --target-kl 0.5 \
  --admin-batch-size 4 \
  --admin-n-samples 8 \
  --admin-temperature 0.7 \
  --admin-top-p 0.95 \
  --c-grid 0.5,1 \
  --out outputs/20260705_credulous_skeptical_qwen3_14b_strict25_sspace_allinstr_n8 \
  "$@"
