#!/usr/bin/env bash
# pull Qwen3-4B walks/answers/calib from the Modal volume, then Jev-judge and render dev results (exclude contaminated layer-0 k/v runs)
set -euo pipefail
cd "$(dirname "$0")/../.."
D=Qwen--Qwen3-4B-g7c7712c6
for sub in walks answers calib vectors; do mkdir -p outputs/bsbench/$D/$sub; uv run --extra benchmark modal volume get --force steering-lite-bsbench-v3 bsbench/$D/$sub outputs/bsbench/$D/ >/dev/null; done
set -a; . ./.env; set +a
cd scripts/bsbench
uv run --extra benchmark python judge.py --cohort dev --refresh --model Qwen/Qwen3-4B 2>&1 | tail -3
uv run --extra benchmark python results.py --cohort dev --model-dir ../../outputs/bsbench/$D --out ../../outputs/bsbench/results/q3-4b-dev --exclude "${EXCLUDE:-value_steer,key_steer}" 2>&1 | tail -2
