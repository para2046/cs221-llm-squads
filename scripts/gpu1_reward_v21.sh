#!/usr/bin/env bash
# GPU 1 (job 1990335) after the reward-v1 run: self-play GRPO with reward v2.1 (shaped21 = v2 + engagement-sync,
# + adv-sd-floor 0.5); everything else identical to selfplay_grpo_v2 (v1) and selfplay_rv2 (v2).
cd ~/cs221/project
while pgrep -f "cluster_selfplay_v2\.sh" >/dev/null; do sleep 60; done
sleep 30
J=1990335
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8395 8 \
   --opponent external --brief facts --channel struct --reward shaped21 > logs/cl_env_rv21.log 2>&1 &
S1=$!
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 8 scripts/cluster_env_server.sh 8396 8 \
   --opponent builtin --brief facts --channel struct > logs/cl_env_rv21_eval.log 2>&1 &
S2=$!
until grep -q "env server ready" logs/cl_env_rv21.log && grep -q "env server ready" logs/cl_env_rv21_eval.log; do
  sleep 5; kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { echo "server failed"; exit 1; }; done
echo "$(date +%H:%M) start selfplay_rv21"
srun --jobid $J --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh selfplay_rv21 http://wheat-05:8395 \
   --mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens 64 --batch 4 --grad-accum 32 \
   --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://wheat-05:8396 \
   --adv-sd-floor 0.5 --out runs/selfplay_rv21 --iters 20 --group-size 4 --episodes-per-iter 8 --save-every 2 \
   --eval-every 4 --eval-episodes 16
echo "$(date +%H:%M) done"; kill $S1 $S2
