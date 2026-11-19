#!/usr/bin/env bash
# Option 1 (user-approved 2026-11-19): continue MAPPO from round 23 with a bigger step on the next free GPU.
# Bigger step: 2 PPO passes, lr 2e-5, up to 2048 decisions per update (was 1 pass, 1e-5, 1024).
# Free GPU = a running hold job that is not training (oat-01/02) and not used by the zero-shot eval.
cd ~/cs221/project
BUSY="1992889 1998007"
while true; do
  U=$(grep -o "using job [0-9]*" logs/f64_untrained_when_gpu.out 2>/dev/null | awk '{print $3}')
  udone=$(grep -c done logs/f64_untrained_when_gpu.out 2>/dev/null)
  for j in $(squeue -h -u $USER -p gpu -t R -o "%i"); do
    case " $BUSY " in *" $j "*) continue;; esac
    [ -z "$U" ] && continue                         # let the zero-shot eval take the first free GPU
    [ "$j" = "$U" ] && [ "$udone" = "0" ] && continue
    echo "$(date +%H:%M) using job $j"
    srun --jobid $j --overlap --ntasks 1 scripts/cluster_mappo.sh mappo_v2 http://barley-03:8473 --algo mappo \
      --adapter runs/mappo_full/adapter_it23 --ref-adapter runs/awr_c2/adapter --critic-init runs/mappo_full/critic_it23.pt \
      --start-iter 23 --iters 24 --out runs/mappo_v2 --episodes-per-iter 16 --max-decisions 2048 --ppo-epochs 2 --lr 2e-5 \
      --eval-every 2 --eval-episodes 16 --eval-scenarios 100000 --save-every 1 --save-episodes
    echo "$(date +%H:%M) done"; exit 0
  done
  sleep 120
done
