#!/usr/bin/env bash
# Zero-shot (untrained) Gemma 4 E4B, C2, on the same 64 new maps (200000-200063), same eval as e2u_4b_struct.
# Runs on the first GPU hold job that is not already busy (oat-01 / oat-02 are training).
cd ~/cs221/project
BUSY="1992889 1998007"
while true; do
  for j in $(squeue -h -u $USER -p gpu -t R -o "%i"); do
    case " $BUSY " in *" $j "*) continue;; esac
    echo "$(date +%H:%M) using job $j"
    srun --jobid $j --overlap --ntasks 1 scripts/cluster_train.sh f64_untrained http://barley-03:8471 --mode eval \
      --out runs/f64_untrained --iters 8 --episodes-per-iter 8 --temperature 0.7 --max-new-tokens 64 \
      --eval-scenarios 200000 --save-episodes
    echo "$(date +%H:%M) done"; exit 0
  done
  sleep 120
done
