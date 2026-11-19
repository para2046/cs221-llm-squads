#!/usr/bin/env bash
# Render the four demo3 games, 1 frame per game loop at 16 fps (1 env step = 8 loops = 0.5 s), sequentially on GPU1.
cd ~/cs221/project; M=external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map; O=artifacts/battle_replay/demo3_video
mkdir -p $O
for spec in "a:runs/demo3v/none/replays/scen100046_11/scen100046_lost_s52_ep0.SC2Replay" \
            "b:runs/demo3v/struct/replays/scen100046_27/scen100046_won_s44_ep0.SC2Replay" \
            "c:runs/demo_untrained/replays/env2/scen100046_lost_s32_103743.SC2Replay" \
            "d:runs/demo_awr/replays/env2/scen100046_won_s45_104443.SC2Replay"; do
  k=${spec%%:*}; f=${spec#*:}
  srun --jobid 1990335 --overlap --ntasks 1 env SC2PATH=$PWD/external/sc2/StarCraftII .venv-smac/bin/python \
    scripts/render_replay_sync.py $f $O/$k.mp4 --map $M --camera fixed --zoom 62 --size 1080x1080 --step-mul 1 --fps 16 --quality ${Q:-6} 2>&1 | grep -E "wrote|Error|Trace"
done
ls -la $O
