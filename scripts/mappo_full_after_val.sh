#!/usr/bin/env bash
# Start the full MAPPO run on GPU1 as soon as the 4-round validation (runs/mappo_val) has finished.
cd ~/cs221/project
P=$(cat logs/mappo_val_train.pid)
while kill -0 $P 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M) validation finished; starting mappo_full"
: > logs/mappo_full.log
srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_mappo.sh mappo_full http://oat-01:8452 --algo mappo \
  --adapter runs/awr_c2/adapter --out runs/mappo_full --iters 40 --episodes-per-iter 16 --max-decisions 1024 \
  --critic-warmup 1 --eval-first --eval-every 2 --eval-episodes 16 --save-every 1 --save-episodes
echo "$(date +%H:%M) mappo_full exited"
