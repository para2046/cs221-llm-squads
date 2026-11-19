#!/usr/bin/env bash
# D1/D2 pilot on the CS221 API: per model, 10 games each: vs built-in (facts, no comm), vs built-in (facts, struct), self-play (facts, struct).
cd ~/cs221/project
export LLM_RPM=11   # 10 workers x 11 = 110 requests/min, under the API's 120/min
U=https://cs221-proxy-stanford.northcentralus.cloudapp.azure.com/v1
for M in openai/gpt-oss-120b thinkingmachines/Inkling-2-Small; do
  t=$(echo $M | sed 's#.*/##; s/[^A-Za-z0-9]//g' | tr A-Z a-z)
  C="--api-key-env CS221_API_KEY --reasoning-effort low --temperature 0.7 --brief facts"
  scripts/collect_play.sh d12_${t}_builtin_none 10 1 --a "llm:$U:$M" --opponent builtin --channel none $C
  scripts/collect_play.sh d12_${t}_builtin_struct 10 1 --a "llm:$U:$M" --opponent builtin --channel struct $C
  scripts/collect_play.sh d12_${t}_self_struct 10 1 --a "llm:$U:$M" --opponent "llm:$U:$M" --channel struct $C   # (self-play: both teams share the worker's budget)
done
