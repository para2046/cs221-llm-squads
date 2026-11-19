#!/usr/bin/env bash
# Evaluate one adapter (no base run). usage: <out_prefix> <temperature> <iters x 8 games> <adapter>
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; TEMP=$2; ITERS=$3; A=$4
NAME="$(basename "$(dirname "$A")")_$(basename "$A")"
srun --jobid=${GPU_JOB:-1971669} --overlap --ntasks=1 --cpus-per-task=2 \
  env HF_HUB_OFFLINE=1 .venv-train/bin/python -m smac_llm.train --mode eval --adapter "$A" \
  --out runs/${PREFIX}_$NAME --env-url http://wheat-05:8390 --env-token-file data/env_token \
  --iters $ITERS --episodes-per-iter 8 --temperature $TEMP --save-episodes > logs/${PREFIX}_$NAME.log 2>&1
echo "== $NAME"; grep '^{' logs/${PREFIX}_$NAME.log
