#!/usr/bin/env bash
# Serve a Gemma model with vLLM inside a running GPU Slurm job.
# usage: srun --jobid=<gpu job> --overlap smac_llm/serve_gemma.sh google/gemma-4-E4B-it gemma4b 8231
set -euo pipefail
MODEL=${1:-google/gemma-4-E4B-it}; NAME=${2:-gemma4b}; PORT=${3:-8231}
PY=/home/users/para2046/projects/202610/multiagent/.venv-vllm/bin/python
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_CACHE_ROOT=/tmp/$USER-vllm-cache
echo "host=$(hostname) port=$PORT model=$MODEL"
exec $PY -m vllm.entrypoints.openai.api_server --model "$MODEL" --served-model-name "$NAME" \
  --host 0.0.0.0 --port "$PORT" --api-key "$(cat /home/users/para2046/projects/202610/multiagent/data/llm_api_key)" \
  --max-model-len 8192 --gpu-memory-utilization 0.85 --language-model-only
