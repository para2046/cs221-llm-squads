#!/usr/bin/env bash
# Reward v2.1 self-play run: starts on whichever GPU frees up first -- GPU 1 (job 1990335) once the reward-v1 run ends,
# or any newly granted cs221 GPU hold job. Env servers run on the extra CPU job (barley-02).
cd ~/cs221/project
CPUJ=1992888
pick() {
  pgrep -f "cluster_selfplay_v2\.sh" >/dev/null || { echo 1990335; return; }
  squeue -h -u $USER -t R -o "%i %j" | awk '/cs221-gpu/ && $1!=1990335 && $1!=1991191 {print $1; exit}'
}
J=""; while [ -z "$J" ]; do J=$(pick); [ -z "$J" ] && sleep 60; done
echo "$(date +%H:%M) using GPU job $J"; sleep 30
srun --jobid $CPUJ --overlap --ntasks 1 --cpus-per-task 14 scripts/cluster_env_server.sh 8399 8 \
   --opponent external --brief facts --channel struct --reward shaped21 > logs/cl_env_rv21.log 2>&1 &
S1=$!
srun --jobid $CPUJ --overlap --ntasks 1 --cpus-per-task 9 scripts/cluster_env_server.sh 8400 8 \
   --opponent builtin --brief facts --channel struct > logs/cl_env_rv21_eval.log 2>&1 &
S2=$!
until grep -q "env server ready" logs/cl_env_rv21.log && grep -q "env server ready" logs/cl_env_rv21_eval.log; do
  sleep 5; kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { echo "server failed"; exit 1; }; done
H=$(grep -o "host=[^ ]*" logs/cl_env_rv21.log | cut -d= -f2)
echo "$(date +%H:%M) start selfplay_rv21 (env on $H)"
srun --jobid $J --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh selfplay_rv21 http://$H:8399 \
   --mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens 64 --batch 4 --grad-accum 32 \
   --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://$H:8400 \
   --adv-sd-floor 0.5 --out runs/selfplay_rv21 --iters 20 --group-size 4 --episodes-per-iter 8 --save-every 2 \
   --eval-every 4 --eval-episodes 16
echo "$(date +%H:%M) done"; kill $S1 $S2
