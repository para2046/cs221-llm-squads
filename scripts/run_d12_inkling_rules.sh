#!/usr/bin/env bash
# After gpt-oss rules cells: Inkling-2-Small (best pilot model) with rules-only brief (no tactic examples), none vs struct, 10 games each.
cd ~/cs221/project
while pgrep -f "scripts/run_d12_rules\.sh" >/dev/null; do sleep 60; done
export LLM_RPM=11
U=https://cs221-proxy-stanford.northcentralus.cloudapp.azure.com/v1
C="--api-key-env CS221_API_KEY --reasoning-effort low --temperature 0.7 --brief rules"
scripts/collect_play.sh d12_inkling2small_rules_none 10 1 --a "llm:$U:thinkingmachines/Inkling-2-Small" --opponent builtin --channel none $C
scripts/collect_play.sh d12_inkling2small_rules_struct2 10 1 --a "llm:$U:thinkingmachines/Inkling-2-Small" --opponent builtin --channel struct $C
