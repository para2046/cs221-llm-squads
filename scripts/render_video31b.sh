#!/usr/bin/env bash
cd ~/cs221/project; M=external/sc2/StarCraftII/Maps/SMAC_Maps
R="srun --jobid 1990335 --overlap --ntasks 1 env SC2PATH=$PWD/external/sc2/StarCraftII .venv-smac/bin/python scripts/render_replay.py"
for c in none struct; do for s in 8007 7007 1007; do
  f=$(ls runs/video31b_$c/replays/seed$s/*.SC2Replay | head -1)
  $R $f artifacts/videos/v31b_${c}_seed$s.mp4 --map $M/32x32_flat.SC2Map --camera fixed --zoom 62 --size 1080x1080 --step-mul 2 2>&1 | grep -E "wrote|Error|Trace"
done; done; echo done
