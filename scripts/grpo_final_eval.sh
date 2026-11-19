#!/usr/bin/env bash
# At ~04:40 wait for the next GRPO checkpoint, stop GRPO training, then run the 64-game eval on GPU2.
cd ~/cs221/project
until [ "$(date +%H%M)" -ge 0440 ] && [ "$(date +%H%M)" -lt 1200 ]; do sleep 60; done
last=$(ls -d runs/grpo_full/adapter_it* | sort -V | tail -1)
echo "$(date +%T) waiting for checkpoint after $last"
while [ "$(ls -d runs/grpo_full/adapter_it* | sort -V | tail -1)" = "$last" ]; do sleep 30; done
sleep 60  # let the save finish
ck=$(ls -d runs/grpo_full/adapter_it* | sort -V | tail -1)
pid=$(cat logs/grpo_full_train.pid); echo "$(date +%T) stopping GRPO (srun pid $pid), evaluating $ck"
kill -TERM "$pid"; sleep 90
srun --jobid 1991191 --overlap --ntasks 1 scripts/cluster_mappo.sh f64_grpo http://barley-02:8462 --algo grpo \
  --adapter "$ck" --out runs/f64_grpo --iters 0 --eval-first --eval-episodes 64 --eval-scenarios 200000 --save-episodes
echo "$(date +%T) done"
