#!/usr/bin/env bash
# After the current GRPO run frees the GPU: behaviour-clone Gemma-31B's best games, eval it, then GRPO from it.
cd ~/cs221/project
G="srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh"
while pgrep -f "out runs/rl_grpo_v1_resume" >/dev/null; do sleep 60; done
echo "$(date +%H:%M) GPU free, BC on teacher wins"
$G bc_win_31b http://wheat-05:8391 --mode bc --data runs/teacher_gemma31b --weighting filter --keep-frac 0.4 \
   --epochs 2 --lr 1e-4 --batch 8 --out runs/bc_win_31b || exit 1
echo "$(date +%H:%M) eval"
$G eval_bc_win_31b http://wheat-05:8391 --mode eval --adapter runs/bc_win_31b/adapter --out runs/eval_bc_win_31b \
   --iters 5 --episodes-per-iter 8 --max-decisions 1024 --temperature 0.7 --save-episodes
echo "$(date +%H:%M) GRPO from BC"
$G rl_grpo_from_win http://wheat-05:8391 --mode online --algo grpo --adapter runs/bc_win_31b/adapter --out runs/rl_grpo_from_win \
   --iters 12 --group-size 4 --episodes-per-iter 16 --max-decisions 1024 --batch 16 --grad-accum 8 --ppo-epochs 1 \
   --lr 3e-5 --save-every 3 --save-episodes --temperature 1.0
echo "$(date +%H:%M) done"
