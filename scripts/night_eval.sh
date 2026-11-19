#!/usr/bin/env bash
# Morning results (2026-11-18): for AWR and DAgger adapters -> 64 games vs built-in AI (scenarios 100000.., paired
# with untrained e2u_4b_struct) and head-to-head vs untrained 4B (8 as A + 8 as B, scenarios 200000..).
cd ~/cs221/project
srv() {  # jobid port cpus log opponent
  srun --jobid $1 --overlap --ntasks 1 --cpus-per-task $3 scripts/cluster_env_server.sh $2 8 --opponent $5 \
    --brief unified --channel struct > $4 2>&1 &
  S=$!; until grep -q "env server ready" $4; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed $4"; exit 1; }; done
  H=$(grep -o "host=[^ ]*" $4 | cut -d= -f2)
}
h2h() {  # gpujob name adapter
  until [ -s logs/h2h_host ]; do sleep 30; done; HH=$(cat logs/h2h_host)
  exec 9>logs/h2h.lock; flock 9
  for T in B A; do
    echo "$(date +%H:%M) h2h $2 base$T"
    srun --jobid $1 --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh h2h_$2_base$T http://$HH:8415 --mode eval \
      --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $3 --iters 1 --episodes-per-iter 8 \
      --eval-scenarios 200000 --base-team $T --out runs/h2h_$2_base$T
  done
  flock -u 9
}
case $1 in
h2hserver)
  until grep -q "collect done" logs/d31b_collect.out; do sleep 30; done
  srv 1992888 8415 12 logs/cl_env_h2h_night.log external; echo $H > logs/h2h_host
  echo "$(date +%H:%M) h2h server on $H"; wait;;
awr)
  until grep -q "AWR eval done" logs/awr_c2.out; do sleep 60; done
  h2h 1990335 awr runs/awr_c2/adapter; echo "$(date +%H:%M) awr all done";;
dagger)
  until grep -q "DAgger done" logs/dagger_c2.out; do sleep 60; done
  A=runs/dagger_c2/adapter_it$(ls -d runs/dagger_c2/adapter_it* | sed "s/.*adapter_it//" | sort -n | tail -1)
  srv 1991191 8412 8 logs/cl_env_dagger_eval.log builtin
  echo "$(date +%H:%M) DAgger eval $A"
  srun --jobid 1991191 --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh e4_dagger_c2 http://$H:8412 --mode eval \
    --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A --iters 8 --episodes-per-iter 8 \
    --eval-scenarios 100000 --out runs/e4_dagger_c2
  kill $S; h2h 1991191 dagger $A; echo "$(date +%H:%M) dagger all done";;
esac
