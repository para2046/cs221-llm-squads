#!/usr/bin/env bash
# SC2 env server on the CPU hold node (run via srun --overlap into the CPU job). usage: cluster_env_server.sh PORT N_ENVS [server args...]
cd ~/cs221/project; port=$1; n=$2; shift 2
exec env SC2PATH=$PWD/external/sc2/StarCraftII .venv-smac/bin/python -m smac_llm.env_server --n-envs $n --port $port --token-file data/env_token "$@"
