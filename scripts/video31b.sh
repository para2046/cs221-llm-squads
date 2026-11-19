#!/usr/bin/env bash
# 31B C0 vs C2 demo games with SC2 replays (seeds where E1 had C2 win, C0 loss).
cd ~/cs221/project
set -a; source .env; set +a
for c in none struct; do mkdir -p runs/video31b_$c/replays logs/video31b; for s in 1007 7007 8007; do
  srun --jobid=1992888 --overlap --ntasks=1 --cpus-per-task=2 \
    env SC2PATH=$PWD/external/sc2/StarCraftII OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    .venv-smac/bin/python -m smac_llm.play --episodes 1 --seed $s --out runs/video31b_$c/seed$s.jsonl \
    --replay-dir $PWD/runs/video31b_$c/replays/seed$s \
    --a "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" --opponent builtin --brief unified --channel $c \
    --api-key-env OPENROUTER_API_KEY --temperature 0.7 > logs/video31b/${c}_$s.log 2>&1 &
done; done
wait; echo done
