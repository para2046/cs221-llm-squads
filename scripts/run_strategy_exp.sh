#!/usr/bin/env bash
# Rule bots vs built-in AI, then Gemma 4 31B with the strategy brief; same seeds as runs/teacher_gemma31b (first 4 eps/worker).
cd ~/cs221/project
scripts/collect.sh rule_scripted 10 4 --policy scripted
scripts/collect.sh rule_kite 10 4 --policy kite
scripts/collect.sh teacher31b_strategy 10 4 --policy llm --base-url https://openrouter.ai/api/v1 \
  --model google/gemma-4-31b-it --api-key-env OPENROUTER_API_KEY --temperature 0.7 --strategy
