#!/usr/bin/env bash
# Main API result: Inkling-2-Small vs built-in AI, 40 games per cell (10 workers x 4, seeds 1000*i+7),
# brief {facts, rules} x channel {none, struct (talk-then-fire v2)}. gpt-oss dropped (user, 2026-11-17).
cd ~/cs221/project
export LLM_RPM=11
U=https://cs221-proxy-stanford.northcentralus.cloudapp.azure.com/v1
M="llm:$U:thinkingmachines/Inkling-2-Small"
C="--api-key-env CS221_API_KEY --reasoning-effort low --temperature 0.7 --opponent builtin"
for cell in "facts none" "facts struct" "rules none" "rules struct"; do
  set -- $cell
  echo "$(date +%H:%M) start ink40_$1_$2"
  scripts/collect_play.sh ink40_$1_$2 10 4 --a "$M" --brief $1 --channel $2 $C
  echo "$(date +%H:%M) done ink40_$1_$2"
done
