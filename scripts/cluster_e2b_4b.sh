#!/usr/bin/env bash
# E2b: untrained 4B vs built-in AI, 40 games per cell: rules-only brief x {none, info, struct}, and facts x struct,
# with the new "talk, then fire together" struct prompt and 128 output tokens when talking.
cd ~/cs221/project
cell() {  # name, max_new_tokens, env-server args
  name=$1; mnt=$2; shift 2
  log=logs/cl_env_$name.log
  srun --jobid 1989631 --overlap --ntasks 1 --cpus-per-task 12 scripts/cluster_env_server.sh 8394 8 "$@" > $log 2>&1 &
  sp=$!
  until grep -q "env server ready" $log; do sleep 5; kill -0 $sp 2>/dev/null || { echo "server failed: $name"; return; }; done
  echo "$(date +%H:%M) start $name"
  srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh $name http://wheat-05:8394 --mode eval \
     --out runs/$name --iters 5 --episodes-per-iter 8 --max-decisions 1024 --temperature 0.7 --max-new-tokens $mnt --save-episodes
  echo "$(date +%H:%M) done $name: $(grep -h '^rollout' logs/$name.log | tail -5 | python3 -c "
import sys,json; r=[json.loads(l[8:]) for l in sys.stdin]; n=sum(x['episodes'] for x in r)
print('win', round(sum(x['win_rate']*x['episodes'] for x in r)/n,3), 'return', round(sum(x['return']*x['episodes'] for x in r)/n,2), 'n', n)")"
  kill $sp; sleep 10
}
cell e2_4b_rules_none    64  --brief rules --channel none
cell e2_4b_rules_info    64  --brief rules --channel info
cell e2_4b_rules_struct2 128 --brief rules --channel struct
cell e2_4b_facts_struct2 128 --brief facts --channel struct
echo "$(date +%H:%M) all done"
