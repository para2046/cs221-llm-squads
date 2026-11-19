#!/usr/bin/env bash
# API models (EVAL ONLY, never training data) on the fixed maps 200000-200039, the same maps the 4B models
# played in runs/f64_*. usage: fixed40_api.sh TAG MODEL_SPEC WORKERS [play args...]
cd ~/cs221/project; set -a; . ./.env; set +a
TAG=$1; SPEC=$2; W=$3; shift 3; N=${N:-40}; START=${START:-200000}; E=$((N / W))
mkdir -p runs/$TAG logs/$TAG
for i in $(seq 0 $((W-1))); do
  srun --jobid=1998155 --overlap --ntasks=1 --cpus-per-task=1 env SC2PATH=$PWD/external/sc2/StarCraftII \
    OPENROUTER_API_KEY=$OPENROUTER_API_KEY .venv-smac/bin/python -m smac_llm.play --episodes $E --seed $((1000*i+7)) \
    --scenario-start $((START + E*i)) --out runs/$TAG/w$i.jsonl --a "$SPEC" --opponent builtin \
    --brief unified --channel struct --api-key-env OPENROUTER_API_KEY --temperature 0.7 "$@" > logs/$TAG/w$i.log 2>&1 &
done
wait; echo "$(date +%H:%M) $TAG done"
