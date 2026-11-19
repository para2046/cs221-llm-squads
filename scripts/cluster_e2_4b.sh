#!/usr/bin/env bash
# E2: Gemma 4 E4B (untrained) vs built-in AI, 40 games per cell, on the held GPU; env servers on the CPU job.
cd ~/cs221/project
cell() {  # name, then env-server args
  name=$1; shift
  log=logs/cl_env_$name.log
  srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8394 8 "$@" > $log 2>&1 &
  sp=$!
  until grep -q "env server ready" $log; do sleep 5; kill -0 $sp 2>/dev/null || { echo "server failed: $name"; return; }; done
  echo "$(date +%H:%M) start $name"
  srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh $name http://wheat-05:8394 --mode eval \
     --out runs/$name --iters 5 --episodes-per-iter 8 --max-decisions 1024 --temperature 0.7 --max-new-tokens 64 --save-episodes
  echo "$(date +%H:%M) done $name: $(grep -h '^rollout' logs/$name.log | tail -5 | python3 -c "
import sys,json; r=[json.loads(l[8:]) for l in sys.stdin]; n=sum(x['episodes'] for x in r)
print('win', round(sum(x['win_rate']*x['episodes'] for x in r)/n,3), 'return', round(sum(x['return']*x['episodes'] for x in r)/n,2), 'n', n)")"
  kill $sp; sleep 10
}
cell e2_4b_none_none    --brief none  --channel none
cell e2_4b_facts_none   --brief facts --channel none
cell e2_4b_facts_vision --brief facts --channel vision
cell e2_4b_facts_struct --brief facts --channel struct
echo "$(date +%H:%M) all done"
