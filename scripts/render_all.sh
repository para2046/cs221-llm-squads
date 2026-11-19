#!/usr/bin/env bash
# Render every demo replay with the real SC2 renderer on the held GPU (EGL needs a GPU node).
cd ~/cs221/project; M=external/sc2/StarCraftII/Maps/SMAC_Maps
R="srun --jobid 1990335 --overlap --ntasks 1 env SC2PATH=$PWD/external/sc2/StarCraftII .venv-smac/bin/python scripts/render_replay.py"
i=0
for spec in "runs/versus_or/replays_gq:32x32_flat_2p" "runs/versus_or/replays_qg:32x32_flat_2p" "runs/replays_sc2:32x32_flat"; do
  d=${spec%%:*}; m=${spec##*:}
  for f in $(ls $d/*.SC2Replay | sort); do i=$((i+1))
    $R $f artifacts/videos/game$i.mp4 --map $M/$m.SC2Map --camera fixed --zoom 62 --size 1080x1080 --step-mul 2 2>&1 | grep -E "wrote|Error|Trace"
  done
done
