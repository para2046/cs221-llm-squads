#!/usr/bin/env bash
# User request 2026-11-19: pause GRPO-vs-AI after round 15 is saved, run the two zero-shot 4B tests (with / without
# history) on the same 64 maps on oat-01, then resume GRPO from round 16 (same settings, 23 rounds in total).
cd ~/cs221/project
J=1992889; SH=3572860; PY=2141175
until [ -f runs/grpo_ai_full/adapter_it15/adapter_model.safetensors ]; do sleep 30; done
sleep 60                                                   # let the save finish
kill $SH; srun --jobid $J --overlap --ntasks 1 kill $PY; sleep 30
echo "$(date +%H:%M) GRPO paused after round 15"
echo "$(date +%H:%M) using job $J" >> logs/f64_untrained_when_gpu.out
srun --jobid $J --overlap --ntasks 1 scripts/cluster_train.sh f64_untrained http://barley-03:8471 --mode eval \
  --out runs/f64_untrained --iters 8 --episodes-per-iter 8 --temperature 0.7 --max-new-tokens 64 \
  --eval-scenarios 200000 --save-episodes
srun --jobid $J --overlap --ntasks 1 scripts/cluster_train.sh f64_untrained_nohist http://barley-03:8474 --mode eval \
  --out runs/f64_untrained_nohist --iters 8 --episodes-per-iter 8 --temperature 0.7 --max-new-tokens 64 \
  --eval-scenarios 200000 --save-episodes
echo "$(date +%H:%M) done" >> logs/f64_untrained_when_gpu.out
echo "$(date +%H:%M) zero-shot tests done; resuming GRPO"
srun --jobid $J --overlap --ntasks 1 scripts/cluster_mappo.sh grpo_ai_full http://barley-03:8471 --algo grpo \
  --adapter runs/grpo_ai_full/adapter_it15 --ref-adapter runs/awr_c2/adapter --out runs/grpo_ai_full --start-iter 16 \
  --iters 7 --episodes-per-iter 16 --group-size 4 --max-decisions 1024 \
  --eval-every 2 --eval-episodes 16 --eval-scenarios 100000 --save-every 1 --save-episodes
ck=$(ls -d runs/grpo_ai_full/adapter_it* | sort -V | tail -1)
echo "$(date +%H:%M) training done; final test on $ck"
srun --jobid $J --overlap --ntasks 1 scripts/cluster_mappo.sh f64_grpo_ai http://barley-03:8471 --algo grpo \
  --adapter "$ck" --out runs/f64_grpo_ai --iters 0 --eval-first --eval-episodes 64 --eval-scenarios 200000 --save-episodes
echo "$(date +%H:%M) done"
