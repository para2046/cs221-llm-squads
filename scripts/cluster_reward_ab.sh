#!/usr/bin/env bash
# After cluster_win_chain.sh: GRPO from the same BC start with the shaped reward (A/B vs rl_grpo_from_win).
cd ~/cs221/project
while pgrep -f "scripts/cluster_win_chain\.sh" >/dev/null; do sleep 60; done
[ -d runs/bc_win_31b/adapter ] || { echo "no BC adapter, stopping"; exit 1; }
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8393 8 --reward shaped > logs/cl_env_shaped.log 2>&1 &
until grep -q "env server ready" logs/cl_env_shaped.log; do sleep 5; done
echo "$(date +%H:%M) GRPO shaped"
srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh rl_grpo_from_win_shaped http://wheat-05:8393 --mode online --algo grpo \
   --adapter runs/bc_win_31b/adapter --out runs/rl_grpo_from_win_shaped --iters 12 --group-size 4 --episodes-per-iter 16 \
   --max-decisions 1024 --batch 16 --grad-accum 8 --ppo-epochs 1 --lr 3e-5 --save-every 3 --save-episodes --temperature 1.0
echo "$(date +%H:%M) done"
