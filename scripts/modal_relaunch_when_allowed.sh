#!/usr/bin/env bash
# Retry until the Modal spend limit allows a launch, then start the priority runs (≤5 at a time).
set -uo pipefail
cd "$(dirname "$0")/.."
M=".venv-modal/bin/modal run --detach smac_llm/modal_app.py::main"
COMMON="--group-size 4 --episodes-per-iter 16 --max-decisions 1024 --batch 16 --grad-accum 8 --ppo-epochs 1 --lr 3e-5 --save-every 3 --save-episodes --temperature 1.0"
try() { name=$1; shift; $M --name $name "$@" > logs/launch_$name.out 2>&1; ! grep -qi "spend limit\|^Error" logs/launch_$name.out; }
until try rl_grpo_v1_resume --args "--mode online --algo grpo --adapter runs/rl_grpo_v1/adapter_it3 --out runs/rl_grpo_v1_resume --iters 9 $COMMON"; do
  echo "$(date +%H:%M) still over the spend limit"; sleep 600
done
echo "$(date +%H:%M) Modal accepts jobs again; launching priority runs"
try rl_dapo_v1_resume --args "--mode online --algo dapo --adapter runs/rl_dapo_v1/adapter_it3 --out runs/rl_dapo_v1_resume --iters 9 $COMMON" &
try rl_ppo_v3_resume --args "--mode online --algo ppo --adapter runs/rl_ppo_v3/adapter_it6 --out runs/rl_ppo_v3_resume --iters 6 --episodes-per-iter 16 --max-decisions 1024 --batch 16 --grad-accum 8 --ppo-epochs 1 --lr 3e-5 --save-every 3 --save-episodes --temperature 1.0" &
scripts/modal_teacher_chain.sh > logs/modal_teacher_chain.out 2>&1 &
wait
echo "$(date +%H:%M) priority 1-2 launched/finished"
