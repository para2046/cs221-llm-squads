#!/usr/bin/env bash
# E4 / E4b with the round-8 adapters of v1, v2, v2.1 (training stopped at it8, agreed 2026-11-17 19:30).
# usage: e4_it8.sh e4  -> GPU1: each adapter, 64 games vs built-in AI, scenarios 100000.. (untrained baseline = e2u_4b_struct)
#        e4_it8.sh h2h -> GPU2: each adapter vs the untrained model, 16 games with trained as A + 16 as B, scenarios 200000..
cd ~/cs221/project
MODE=$1; CJ=1992888
declare -A AD=([u_v1]=runs/u_v1/adapter_it8 [u_v2]=runs/u_v2/adapter_it8 [u_v21]=runs/u_v21_modal/u_v21/adapter_it8)
if [ "$MODE" = e4 ]; then J=1990335; P=8401; SRVARGS="--opponent builtin"; CPUS=8; HOSTJOB=$CJ
else J=1991191; P=8403; SRVARGS="--opponent external"; CPUS=12; HOSTJOB=$CJ; fi
LOG=logs/cl_env_it8_$MODE.log
srun --jobid $HOSTJOB --overlap --ntasks 1 --cpus-per-task $CPUS scripts/cluster_env_server.sh $P 8 $SRVARGS --brief unified --channel struct > $LOG 2>&1 &
S=$!
until grep -q "env server ready" $LOG; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed"; exit 1; }; done
H=$(grep -o "host=[^ ]*" $LOG | cut -d= -f2)
G="srun --jobid $J --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh"
for N in u_v1 u_v2 u_v21; do
  EV="--mode eval --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter ${AD[$N]}"
  if [ "$MODE" = e4 ]; then
    echo "$(date +%H:%M) E4 $N"
    $G e4it8_$N http://$H:$P $EV --iters 8 --episodes-per-iter 8 --eval-scenarios 100000 --out runs/e4it8_$N
  else
    for T in B A; do
      echo "$(date +%H:%M) E4b $N base$T"
      $G e4bit8_${N}_base$T http://$H:$P $EV --iters 2 --episodes-per-iter 8 --eval-scenarios 200000 --base-team $T --out runs/e4bit8_${N}_base$T
    done
  fi
done
echo "$(date +%H:%M) all done"; kill $S
