#!/usr/bin/env bash
# Teacher pipeline on Modal: AWR on Gemma-31B games -> evaluate -> GRPO from the AWR adapter.
set -uo pipefail
cd "$(dirname "$0")/.."
M=.venv-modal/bin/modal
COMMON="--group-size 4 --episodes-per-iter 16 --max-decisions 1024 --batch 16 --grad-accum 8 --ppo-epochs 1 --lr 3e-5 --save-every 3 --save-episodes --temperature 1.0"
$M run smac_llm/modal_app.py::main --name bc_awr_31b \
  --args "--mode bc --data runs/teacher_gemma31b --weighting awr --awr-beta 1.0 --epochs 2 --lr 1e-4 --batch 8 --out runs/bc_awr_31b" \
  > logs/launch_bc_awr_31b.out 2>&1
$M run --detach smac_llm/modal_app.py::main --name eval_bc_awr_31b \
  --args "--mode eval --adapter runs/bc_awr_31b/adapter --out runs/eval_bc_awr_31b --iters 5 --episodes-per-iter 8 --temperature 0.7 --save-episodes" \
  > logs/launch_eval_bc_awr_31b.out 2>&1 &
$M run --detach smac_llm/modal_app.py::main --name rl_grpo_from_awr \
  --args "--mode online --algo grpo --adapter runs/bc_awr_31b/adapter --out runs/rl_grpo_from_awr --iters 12 $COMMON" \
  > logs/launch_rl_grpo_from_awr.out 2>&1 &
wait
