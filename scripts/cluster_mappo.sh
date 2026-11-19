#!/usr/bin/env bash
# MAPPO on the held cluster GPU. usage: cluster_mappo.sh NAME ENV_URL [mappo args...]
cd ~/cs221/project; name=$1; url=$2; shift 2
set -a; . ./.env; set +a
exec env HF_HUB_OFFLINE=1 GEN_BS=${GEN_BS:-32} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv-train/bin/python -m smac_llm.mappo "$@" \
  --env-url $url --env-token-file data/env_token >> logs/$name.log 2>&1
