#!/usr/bin/env bash
# One 31B game on a fixed scenario with replay. usage: v3_play.sh CHANNEL SCENARIO TAG
cd ~/cs221/project; set -a; source .env; set +a
c=$1; s=$2; tag=$3; d=runs/demo3v/$c; mkdir -p $d/replays/scen${s}_$tag logs/demo3v
srun --jobid=1992888 --overlap --ntasks=1 --cpus-per-task=2 \
  env SC2PATH=$PWD/external/sc2/StarCraftII OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  .venv-smac/bin/python scripts/play_scenario.py --episodes 1 --seed $tag --scenario $s --spawn-seed ${SPAWN:-0} --out $d/scen${s}_$tag.jsonl \
  --replay-dir $PWD/$d/replays/scen${s}_$tag \
  --a "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" --opponent builtin --brief unified --channel $c \
  --api-key-env OPENROUTER_API_KEY --temperature 0.7 > logs/demo3v/${c}_${s}_$tag.log 2>&1
tail -1 logs/demo3v/${c}_${s}_$tag.log
