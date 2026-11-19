#!/usr/bin/env bash
# E1 (unified protocol): Inkling-2-Small vs built-in AI, C0/C1/C2, 40 games each (10 workers x 4, seeds 1000*i+7),
# CS221 API (rate-limited, so cells run one after another). Optional: Gemma 31B C0/C2 on OpenRouter, in parallel.
cd ~/cs221/project
U=https://cs221-proxy-stanford.northcentralus.cloudapp.azure.com/v1
if [ "$1" = "inkling" ]; then
  export LLM_RPM=11
  for ch in none info struct; do
    echo "$(date +%H:%M) start e1u_inkling_$ch"
    scripts/collect_play.sh e1u_inkling_$ch 10 4 --a "llm:$U:thinkingmachines/Inkling-2-Small" --opponent builtin \
      --brief unified --channel $ch --api-key-env CS221_API_KEY --reasoning-effort low --temperature 0.7
  done
else
  export CPU_JOB=1992888
  for ch in none struct; do
    echo "$(date +%H:%M) start e1u_31b_$ch"
    scripts/collect_play.sh e1u_31b_$ch 10 4 --a "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" --opponent builtin \
      --brief unified --channel $ch --api-key-env OPENROUTER_API_KEY --temperature 0.7
  done
fi
echo "$(date +%H:%M) all done"
