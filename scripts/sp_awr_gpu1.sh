#!/usr/bin/env bash
# GPU1 (job 1990335, ends 18:18): as soon as runs/sp_awr/adapter_it4 exists, evaluate it (64 games vs built-in,
# scenarios 100000) and run h2h vs untrained (8 as A + 8 as B). Env servers on GPU1's own 8 CPUs.
cd ~/cs221/project
G=1990335; A=runs/sp_awr/adapter_it4
srv() {  # port log opponent
  srun --jobid $G --overlap --ntasks 1 --cpus-per-task 6 scripts/cluster_env_server.sh $1 8 --opponent $3 \
    --brief unified --channel struct > $2 2>&1 &
  S=$!; until grep -q "env server ready" $2; do sleep 5; kill -0 $S 2>/dev/null || { echo "server failed $2"; exit 1; }; done
  H=$(grep -o "host=[^ ]*" $2 | cut -d= -f2)
}
until [ -s $A/adapter_model.safetensors ]; do sleep 60; done; sleep 30
srv 8431 logs/cl_env_sp4_eval.log builtin
echo "$(date +%H:%M) eval $A on $H"
srun --jobid $G --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh e4_sp_awr_it4 http://$H:8431 --mode eval \
  --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A --iters 8 --episodes-per-iter 8 \
  --eval-scenarios 100000 --out runs/e4_sp_awr_it4
echo "$(date +%H:%M) eval done"; kill $S; sleep 10
srv 8432 logs/cl_env_sp4_h2h.log external
for T in B A; do
  srun --jobid $G --overlap --ntasks 1 env GEN_BS=32 scripts/cluster_train.sh h2h_sp_awr_it4_base$T http://$H:8432 --mode eval \
    --temperature 0.7 --max-new-tokens 64 --save-episodes --adapter $A --iters 1 --episodes-per-iter 8 \
    --eval-scenarios 200000 --base-team $T --out runs/h2h_sp_awr_it4_base$T
done
echo "$(date +%H:%M) it4 all done"; kill $S
