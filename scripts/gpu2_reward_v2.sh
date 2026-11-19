#!/usr/bin/env bash
# GPU 2 (job 1991191, oat-03): self-play GRPO with reward v2 (shaped2 + adv-sd-floor 0.5), otherwise identical to
# selfplay_grpo_v2 on GPU 1 (same seed / scenario schedule, condition, optimiser, SIL, eval schedule).
cd ~/cs221/project
until grep -q "done sp_eval_base" logs/gpu2_eval.out; do sleep 30; done
kill -- -3975276; sleep 20
J=1991191
srun --jobid $J --overlap --ntasks 1 --cpus-per-task 9 scripts/cluster_env_server.sh 8398 8 \
   --opponent external --brief facts --channel struct --reward shaped2 > logs/cl_env_rv2.log 2>&1 &
S1=$!
srun --jobid $J --overlap --ntasks 1 --cpus-per-task 4 scripts/cluster_env_server.sh 8397 8 \
   --opponent builtin --brief facts --channel struct > logs/cl_env_rv2_eval.log 2>&1 &
S2=$!
until grep -q "env server ready" logs/cl_env_rv2.log && grep -q "env server ready" logs/cl_env_rv2_eval.log; do
  sleep 5; kill -0 $S1 2>/dev/null && kill -0 $S2 2>/dev/null || { echo "server failed"; exit 1; }; done
H=$(grep -o "host=[^ ]*" logs/cl_env_rv2.log | cut -d= -f2)
echo "$(date +%H:%M) start selfplay_rv2 on $H"
srun --jobid $J --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh selfplay_rv2 http://$H:8398 \
   --mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens 64 --batch 4 --grad-accum 32 \
   --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 --eval-env-url http://$H:8397 \
   --adv-sd-floor 0.5 --out runs/selfplay_rv2 --iters 20 --group-size 4 --episodes-per-iter 8 --save-every 2 \
   --eval-every 4 --eval-episodes 16
echo "$(date +%H:%M) done"; kill $S1 $S2
