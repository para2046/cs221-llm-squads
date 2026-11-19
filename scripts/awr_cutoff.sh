#!/usr/bin/env bash
# Stop AWR right after the step-20000 adapter save (time budget for the 2026-11-18 morning results).
cd ~/cs221/project
until grep -qa '"step": 20010,' logs/awr_c2.log; do sleep 30; done
sleep 5; P=$(ps -eo pid,args | grep "srun --jobid 1990335 --overlap --ntasks 1 scripts/cluster_train.sh awr_c2" | grep -v grep | awk '{print $1}' | head -1)
echo "$(date +%H:%M) stopping AWR training at step 20000 (srun pid $P)"; kill $P
