#!/usr/bin/env bash
# 4B vs 4B self-play RL on the held GPU, in the best prompt/comm condition from the E2 cells.
# On-policy GRPO (both teams' decisions, shaped symmetric reward) + off-policy self-imitation of past wins;
# every 3 iterations the current weights play 16 games vs the built-in AI on a second env server.
cd ~/cs221/project
# (v2: condition fixed from v1 selection)
BRIEF=facts; CHAN=struct; MNT=64
echo "$(date +%H:%M) condition: brief=$BRIEF channel=$CHAN max_new_tokens=$MNT"
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8395 8 \
   --opponent external --brief $BRIEF --channel $CHAN --reward shaped > logs/cl_env_selfplay2.log 2>&1 &
S1=$!
srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 8 scripts/cluster_env_server.sh 8396 8 \
   --opponent builtin --brief $BRIEF --channel $CHAN > logs/cl_env_selfplay2_eval.log 2>&1 &
S2=$!
until grep -q "env server ready" logs/cl_env_selfplay2.log && grep -q "env server ready" logs/cl_env_selfplay2_eval.log; do
  sleep 5; kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { echo "server failed"; exit 1; }; done
export GEN_BS=8
G="srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh"
COMMON="--mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens $MNT --batch 4 --grad-accum 32 \
        --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://wheat-05:8396"
echo "$(date +%H:%M) full run"
$G selfplay_grpo_v2 http://wheat-05:8395 $COMMON --out runs/selfplay_grpo_v2 --iters 20 --group-size 4 \
   --episodes-per-iter 8 --save-every 2 --eval-every 4 --eval-episodes 16
echo "$(date +%H:%M) done"; kill $S1 $S2
