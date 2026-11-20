#!/usr/bin/env bash
# Zero-shot (untrained) Gemma 4 E4B, C2, on the same 64 new maps (200000-200063): with the 5-step history (8471)
# and without it (8474, --no-history), back to back on the first free GPU hold job (oat-01 is training).
# mappo_v2_when_gpu.sh waits for the "done" line below before it takes this GPU.
cd ~/cs221/project
BUSY="1992889 1998007"
while true; do
  for j in $(squeue -h -u $USER -p gpu -t R -o "%i"); do
    case " $BUSY " in *" $j "*) continue;; esac
    echo "$(date +%H:%M) using job $j"
    [ -s runs/f64_untrained/eval_episodes.jsonl ] || \
    srun --jobid $j --overlap --ntasks 1 scripts/cluster_train.sh f64_untrained http://barley-03:8471 --mode eval \
      --out runs/f64_untrained --iters 8 --episodes-per-iter 8 --temperature 0.7 --max-new-tokens 64 \
      --eval-scenarios 200000 --save-episodes
    srun --jobid $j --overlap --ntasks 1 scripts/cluster_train.sh f64_untrained_nohist http://barley-03:8474 --mode eval \
      --out runs/f64_untrained_nohist --iters 8 --episodes-per-iter 8 --temperature 0.7 --max-new-tokens 64 \
      --eval-scenarios 200000 --save-episodes
    echo "$(date +%H:%M) done"; exit 0
  done
  sleep 120
done
