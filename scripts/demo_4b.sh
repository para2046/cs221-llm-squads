#!/usr/bin/env bash
# Demo videos: untrained 4B vs AWR 4B on the same C2 scenarios, with SC2 replays (one game per env -> one replay each).
# usage: demo_4b.sh TAG ADAPTER|none SRVJOB CPUS PORT SCEN
cd ~/cs221/project; TAG=$1; AD=$2; SJ=$3; C=$4; P=$5; SC=$6
L=logs/cl_env_demo_$TAG.log; mkdir -p runs/demo_$TAG
srun --jobid $SJ --overlap --ntasks 1 --cpus-per-task $C scripts/cluster_env_server.sh $P 8 --opponent builtin \
  --brief unified --channel struct --replay-dir runs/demo_$TAG/replays > $L 2>&1 &
S=$!; echo "server pid $S"
until grep -q "env server ready" $L; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed $L"; exit 1; }; done
H=$(grep -o "host=[^ ]*" $L | cut -d= -f2)
X=""; [ "$AD" != none ] && X="--adapter $AD"
echo "$(date +%H:%M) eval $TAG on $H:$P"
srun --jobid 1990335 --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh demo_$TAG http://$H:$P --mode eval \
  --temperature 0.7 --max-new-tokens 64 --save-episodes $X --iters 1 --episodes-per-iter 8 \
  --eval-scenarios $SC --out runs/demo_$TAG
echo "$(date +%H:%M) eval done"; sleep 5; kill $S; wait
