#!/usr/bin/env bash
# Play one demo game with an OpenRouter model, with short intent notes, for the replay viewer.
# usage: scripts/demo_game.sh <openrouter model> <out.jsonl> [seed]
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
MODEL=$1; OUT=$2; SEED=${3:-3}
mkdir -p "$(dirname "$OUT")"
srun --jobid=${CPU_JOB:-1989631} --overlap --ntasks=1 --cpus-per-task=2 \
  env SC2PATH=$PWD/external/sc2/StarCraftII OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
  .venv-smac/bin/python -m smac_llm.run --map 10gen_protoss --n-units 5 --policy llm \
  --base-url https://openrouter.ai/api/v1 --model "$MODEL" --api-key-env OPENROUTER_API_KEY \
  --episodes 1 --seed "$SEED" --explain --reasoning-effort low --max-tokens 1500 --out "$OUT"
