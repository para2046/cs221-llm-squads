#!/usr/bin/env bash
# Complete the API models (EVAL ONLY) on maps 200040-200063 so every model has the same 64 fixed maps.
cd ~/cs221/project
N=24 START=200040 scripts/fixed40_api.sh evalonly_31b_fixed64rest "llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it" 8
N=24 START=200040 scripts/fixed40_api.sh evalonly_luna_fixed64rest "llm:https://openrouter.ai/api/v1:openai/gpt-5.6-luna" 8 --reasoning-effort low
N=24 START=200040 scripts/fixed40_api.sh evalonly_gflashlite_fixed64rest "llm:https://openrouter.ai/api/v1:google/gemini-3.5-flash-lite" 8 --reasoning-effort low
echo "$(date +%H:%M) all done"
