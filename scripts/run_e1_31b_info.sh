#!/usr/bin/env bash
cd ~/cs221/project
export CPU_JOB=1992888
echo "$(date +%H:%M) start e1u_31b_info"
scripts/collect_play.sh e1u_31b_info 10 4 --a "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" --opponent builtin \
  --brief unified --channel info --api-key-env OPENROUTER_API_KEY --temperature 0.7
echo "$(date +%H:%M) all done"
