#!/usr/bin/env bash
# Unified protocol, one cluster GPU: E3 self-play training (one reward arm) -> E4 final eval vs built-in AI (64 games,
# fixed scenarios 100000..100063) -> E4b head-to-head vs the untrained model (16 games per side).
# usage: unified_gpu_pipeline.sh NAME GPUJOB CPUJOB PORT0 REWARD [extra train args]
cd ~/cs221/project
NAME=$1; J=$2; CJ=$3; P=$4; R=$5; shift 5; EXTRA="$*"
srv() {  # port log args...
  port=$1; log=$2; shift 2
  srun --jobid $CJ --overlap --ntasks 1 --cpus-per-task ${CPUS:-8} scripts/cluster_env_server.sh $port 8 "$@" > $log 2>&1 &
  echo $!
}
wait_ready() { until grep -q "env server ready" $1; do sleep 5; kill -0 $2 2>/dev/null || { echo "server failed $1"; exit 1; }; done; }
G="srun --jobid $J --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh"
S1=$(CPUS=12 srv $P logs/cl_env_${NAME}.log --opponent external --brief unified --channel struct --reward $R)
S2=$(srv $((P+1)) logs/cl_env_${NAME}_eval.log --opponent builtin --brief unified --channel struct)
wait_ready logs/cl_env_${NAME}.log $S1; wait_ready logs/cl_env_${NAME}_eval.log $S2
H=$(grep -o "host=[^ ]*" logs/cl_env_${NAME}.log | cut -d= -f2)
echo "$(date +%H:%M) E3 train $NAME (reward $R) env on $H"
$G $NAME http://$H:$P --mode online --algo grpo --credit rtg --temperature 1.0 --max-new-tokens 64 --batch 4 \
   --grad-accum 32 --ppo-epochs 1 --lr 3e-5 --max-decisions 1024 --save-episodes --sil-coef 0.1 \
   --eval-env-url http://$H:$((P+1)) --iters 20 --group-size 4 --episodes-per-iter 8 --save-every 2 --eval-every 4 \
   --eval-episodes 16 --out runs/$NAME $EXTRA
A=runs/$NAME/adapter_it$(ls -d runs/$NAME/adapter_it* | sed "s/.*adapter_it//" | sort -n | tail -1); echo "$(date +%H:%M) E3 done, final adapter $A"
EV="--mode eval --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A"
$G e4_$NAME http://$H:$((P+1)) $EV --iters 8 --episodes-per-iter 8 --eval-scenarios 100000 --out runs/e4_$NAME
echo "$(date +%H:%M) E4 done"
kill $S1; sleep 15; S3=$(CPUS=12 srv $P logs/cl_env_${NAME}_h2h.log --opponent external --brief unified --channel struct)
wait_ready logs/cl_env_${NAME}_h2h.log $S3
for T in B A; do   # base-team B: trained plays A; base-team A: trained plays B
  $G e4b_${NAME}_base$T http://$H:$P $EV --iters 2 --episodes-per-iter 8 --eval-scenarios 200000 --base-team $T \
     --out runs/e4b_${NAME}_base$T
done
echo "$(date +%H:%M) all done"; kill $S2 $S3
