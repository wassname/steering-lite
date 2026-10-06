#!/usr/bin/env bash
# PI/OpenAI: backfill cached 9B walks; no extraction or old-dose regeneration.
set -euo pipefail
cd /workspace/2026/lite/steering-lite-bsbench
D=slop/reviews/2026-10-07_low_dose
uv run --extra benchmark modal run scripts/bsbench/run_modal.py::main \
  --cohort full --methods angular_steering,chars,corda_pca,cosine_gated,directional_ablation,linear_act,mean_diff,pca,query_steer,sink_split,sink_split_resid,spherical,sspace,sspace_ablate,sspace_pca,sspace_pool,sspace_scale,topk_clusters,value_gram,vjp_resid,vjp_value \
  --seeds 0,1,2 --extra "--preset qwen3.5-9b --pairs bsbench_v1 --controls" > "$D/walks_learned.log" 2>&1
uv run --extra benchmark modal run scripts/bsbench/run_modal.py::main \
  --cohort full --methods random --seeds auto --extra "--preset qwen3.5-9b --pairs bsbench_v1" > "$D/walks_random.log" 2>&1
printf 'LOW_DOSE_WALKS_COMPLETE\n'
