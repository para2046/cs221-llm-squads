#!/usr/bin/env bash
# Collect SMACv2 episodes in parallel SC2 instances inside the CPU hold job.
# usage: scripts/collect.sh <tag> <workers> <episodes_per_worker> <smac_llm.run args...>
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a
export LLM_KEY=$(cat /home/users/para2046/projects/202610/multiagent/data/llm_api_key)
CPU_JOB=${CPU_JOB:-1989631}
TAG=$1; W=$2; E=$3; shift 3
mkdir -p runs/$TAG logs/$TAG
for i in $(seq 0 $((W-1))); do
  srun --jobid=$CPU_JOB --overlap --ntasks=1 --cpus-per-task=2 \
    env SC2PATH=$PWD/external/sc2/StarCraftII OPENROUTER_API_KEY=$OPENROUTER_API_KEY LLM_KEY=$LLM_KEY \
    .venv-smac/bin/python -m smac_llm.run --episodes $E --seed $((1000*i+7)) \
    --out runs/$TAG/w$i.jsonl "$@" > logs/$TAG/w$i.log 2>&1 &
done
wait
.venv-smac/bin/python scripts/summarize.py runs/$TAG | tee runs/$TAG/summary.json
