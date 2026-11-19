#!/usr/bin/env bash
# GRPO vs the built-in AI (no self-play), same budget/settings as runs/mappo_full, from the SFT adapter.
# Then the fresh-seed 64-game test (200000-200063) on the LAST checkpoint.
cd ~/cs221/project
URL=http://barley-03:8471
srun --jobid 1992889 --overlap --ntasks 1 scripts/cluster_mappo.sh grpo_ai_full $URL --algo grpo \
  --adapter runs/awr_c2/adapter --ref-adapter runs/awr_c2/adapter --out runs/grpo_ai_full \
  --iters 23 --episodes-per-iter 16 --group-size 4 --max-decisions 1024 \
  --eval-every 2 --eval-episodes 16 --eval-scenarios 100000 --save-every 1 --save-episodes
ck=$(ls -d runs/grpo_ai_full/adapter_it* | sort -V | tail -1)
echo "$(date +%H:%M) training done; final test on $ck"
srun --jobid 1992889 --overlap --ntasks 1 scripts/cluster_mappo.sh f64_grpo_ai $URL --algo grpo \
  --adapter "$ck" --out runs/f64_grpo_ai --iters 0 --eval-first --eval-episodes 64 --eval-scenarios 200000 --save-episodes
echo "$(date +%H:%M) done"
