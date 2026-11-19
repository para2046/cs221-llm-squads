#!/usr/bin/env bash
# Second GPU (job 1991191, oat-03): 40-game evals vs built-in AI of the self-play RL checkpoints, same condition
# as training (facts brief, struct channel, 64 tokens, temperature 0.7). Env server runs on the GPU node's own CPUs.
cd ~/cs221/project
J=1991191; P=8397
srun --jobid $J --overlap --ntasks 1 --cpus-per-task 8 scripts/cluster_env_server.sh $P 8 \
   --opponent builtin --brief facts --channel struct > logs/cl_env_gpu2.log 2>&1 &
sp=$!
until grep -q "env server ready" logs/cl_env_gpu2.log; do sleep 5; kill -0 $sp 2>/dev/null || { echo "server failed"; exit 1; }; done
HOST=$(grep -o "host=[^ ]*" logs/cl_env_gpu2.log | cut -d= -f2)
ev() {  # name [extra args]
  name=$1; shift
  echo "$(date +%H:%M) start $name"
  srun --jobid $J --overlap --ntasks 1 env GEN_BS=40 scripts/cluster_train.sh $name http://$HOST:$P --mode eval \
     --out runs/$name --iters 5 --episodes-per-iter 8 --max-decisions 1024 --temperature 0.7 --max-new-tokens 64 \
     --save-episodes "$@"
  echo "$(date +%H:%M) done $name: $(grep -h '^rollout' logs/$name.log | tail -5 | python3 -c "
import sys,json; r=[json.loads(l[8:]) for l in sys.stdin]; n=sum(x['episodes'] for x in r)
print('win', round(sum(x['win_rate']*x['episodes'] for x in r)/n,3), 'return', round(sum(x['return']*x['episodes'] for x in r)/n,2), 'n', n)")"
}
ev sp_eval_base
for it in 4 8 12 16 20; do
  A=runs/selfplay_grpo_v2/adapter_it$it
  until [ -f $A/adapter_model.safetensors ] || ! pgrep -f "cluster_selfplay_v2\.sh" >/dev/null; do sleep 120; done
  [ -f $A/adapter_model.safetensors ] || break
  sleep 30; ev sp_eval_it$it --adapter $A
done
echo "$(date +%H:%M) all done"; kill $sp
