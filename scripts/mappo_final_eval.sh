#!/usr/bin/env bash
# Approved by the user (2026-11-18): ~06:20 wait for the next MAPPO checkpoint, stop MAPPO training, then run the
# fresh-seed 64-game evals (seeds 200000-200063) for the 53% SFT model and the last MAPPO checkpoint on oat-01 (job 1992890).
cd ~/cs221/project
until [ "$(date +%H%M)" -ge 0620 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done
last=$(ls -d runs/mappo_full/adapter_it* | sort -V | tail -1)
echo "$(date +%T) waiting for checkpoint after $last"
while [ "$(ls -d runs/mappo_full/adapter_it* | sort -V | tail -1)" = "$last" ]; do sleep 30; done
sleep 60
ck=$(ls -d runs/mappo_full/adapter_it* | sort -V | tail -1)
pid=$(cat logs/mappo_full_train.pid); echo "$(date +%T) stopping MAPPO (srun pid $pid); final checkpoint $ck"
kill -TERM "$pid"; sleep 90
for spec in "f64_sft runs/awr_c2/adapter" "f64_mappo $ck"; do
  set -- $spec; mkdir -p runs/$1
  echo "$(date +%T) eval $1 ($2)"
  srun --jobid 1992890 --overlap --ntasks 1 scripts/cluster_mappo.sh $1 http://barley-02:8453 --algo mappo \
    --adapter "$2" --out runs/$1 --iters 0 --eval-first --eval-episodes 64 --eval-scenarios 200000
done
echo "$(date +%T) done"
