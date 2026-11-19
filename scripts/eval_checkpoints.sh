#!/usr/bin/env bash
# Evaluate the base model and RL checkpoints on the env server with identical settings.
# usage: scripts/eval_checkpoints.sh <out_prefix> <temperature> <iters x 8 games> [adapter dirs...]
set -uo pipefail
cd "$(dirname "$0")/.."
PREFIX=$1; TEMP=$2; ITERS=$3; shift 3
run() {  # $1 = name, $2 = adapter or ""
  srun --jobid=${GPU_JOB:-1971669} --overlap --ntasks=1 --cpus-per-task=2 \
    env HF_HUB_OFFLINE=1 .venv-train/bin/python -m smac_llm.train --mode eval \
    ${2:+--adapter $2} --out runs/${PREFIX}_$1 --env-url http://wheat-05:8390 \
    --env-token-file data/env_token --iters $ITERS --episodes-per-iter 8 --temperature $TEMP \
    --save-episodes > logs/${PREFIX}_$1.log 2>&1
  echo "== $1"; grep '^{' logs/${PREFIX}_$1.log
}
run base ""
for a in "$@"; do run "$(basename "$(dirname "$a")")_$(basename "$a")" "$a"; done
