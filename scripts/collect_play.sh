#!/usr/bin/env bash
# Parallel smac_llm.play games inside the CPU hold job (same seeds as collect.sh: 1000*i+7).
# usage: scripts/collect_play.sh <tag> <workers> <episodes_per_worker> <smac_llm.play args...>
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
CPU_JOB=${CPU_JOB:-1989631}
TAG=$1; W=$2; E=$3; shift 3
mkdir -p runs/$TAG logs/$TAG
for i in $(seq 0 $((W-1))); do
  srun --jobid=$CPU_JOB --overlap --ntasks=1 --cpus-per-task=2 \
    env LLM_RPM=${LLM_RPM:-0} SC2PATH=$PWD/external/sc2/StarCraftII CS221_API_KEY=$CS221_API_KEY OPENROUTER_API_KEY=$OPENROUTER_API_KEY \
    .venv-smac/bin/python -m smac_llm.play --episodes $E --seed $((1000*(i+${SEED_BASE:-0})+7)) \
    --out runs/$TAG/w$i.jsonl "$@" > logs/$TAG/w$i.log 2>&1 &
done
wait
.venv-smac/bin/python scripts/summarize.py runs/$TAG | tee runs/$TAG/summary.json
