#!/usr/bin/env bash
# Stop sp_awr training when adapter_it6 is saved or at 15:50, whichever first; sp_awr.sh then evaluates the latest
# adapter on GPU2 (64 games + h2h), finishing before ~17:30.
cd ~/cs221/project
until [ -s runs/sp_awr/adapter_it6/adapter_model.safetensors ] || [ $(date +%H%M) -ge 1550 ]; do sleep 60; done
sleep 20
P=$(ps -u $USER -o pid,cmd | awk '/srun --jobid 1991191/ && /cluster_train.sh sp_awr / {print $1; exit}')
echo "$(date +%H:%M) cutoff: killing training srun $P (latest: $(ls -d runs/sp_awr/adapter_it* | tail -1))"
[ -n "$P" ] && kill $P
