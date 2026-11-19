#!/usr/bin/env bash
# Continue self-play RL from the iteration-20 adapter with bigger batches, then evaluate it.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=runs/rl_selfplay_v2_from_it20
srun --jobid=${GPU_JOB:-1971669} --overlap --ntasks=1 --cpus-per-task=2 \
  env HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv-train/bin/python -m smac_llm.train --mode online --adapter runs/rl_selfplay_v1/adapter_it20 \
  --out $OUT --env-url http://wheat-05:8390 --env-token-file data/env_token \
  --iters ${ITERS:-9} --episodes-per-iter 16 --max-decisions 2048 --batch 16 --grad-accum 8 \
  --ppo-epochs 1 --lr 3e-5 --kl 0.05 --save-every 3 --save-episodes > logs/rl_selfplay_v2.log 2>&1
last=$(ls -d $OUT/adapter_it* | sort -V | tail -1)
echo "evaluating $last"
scripts/eval_checkpoints_one.sh eval_t07 0.7 5 "$last"
