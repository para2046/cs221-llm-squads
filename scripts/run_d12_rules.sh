#!/usr/bin/env bash
# After the D1/D2 pilot: gpt-oss-120b with the rules-only brief, vs built-in AI, none vs struct (new prompt), 10 games each.
cd ~/cs221/project
while pgrep -f "scripts/run_d12_pilot\.sh" >/dev/null; do sleep 60; done
export LLM_RPM=11
U=https://cs221-proxy-stanford.northcentralus.cloudapp.azure.com/v1
C="--api-key-env CS221_API_KEY --reasoning-effort low --temperature 0.7 --brief rules"
scripts/collect_play.sh d12_gptoss120b_rules_none 10 1 --a "llm:$U:openai/gpt-oss-120b" --opponent builtin --channel none $C
scripts/collect_play.sh d12_gptoss120b_rules_struct2 10 1 --a "llm:$U:openai/gpt-oss-120b" --opponent builtin --channel struct $C
