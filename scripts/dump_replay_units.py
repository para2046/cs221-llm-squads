"""Dump raw unit states (pos / hp / shield / type / owner) from a .SC2Replay, every game loop step.

Replays the file through the headless SC2 engine (no rendering, CPU only) and writes JSON:
  {"loops": [game_loop, ...], "units": [[{tag, type, owner, x, y, hp, hp_max, sh, sh_max}, ...], ...]}

usage: SC2PATH=external/sc2/StarCraftII .venv-smac/bin/python scripts/dump_replay_units.py REPLAY OUT.json
         [--map external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map] [--step 1]
"""
import argparse
import json
import os
import sys

from absl import flags
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib.remote_controller import Status
from s2clientprotocol import sc2api_pb2 as sc_pb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replay")
    p.add_argument("out")
    p.add_argument("--map", default="external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map")
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--player", type=int, default=1)
    a = p.parse_args()
    flags.FLAGS([sys.argv[0]])
    rc = run_configs.get()
    opts = sc_pb.InterfaceOptions(raw=True, score=False)
    loops, frames = [], []
    with rc.start(want_rgb=False, full_screen=False) as c:
        c.start_replay(sc_pb.RequestStartReplay(replay_data=open(os.path.abspath(a.replay), "rb").read(),
                                                map_data=open(a.map, "rb").read(), options=opts,
                                                observed_player_id=a.player, disable_fog=True))
        while c.status == Status.in_replay:
            obs = c.observe()
            loops.append(obs.observation.game_loop)
            frames.append([{"tag": u.tag, "type": u.unit_type, "owner": u.owner, "x": round(u.pos.x, 3),
                            "y": round(u.pos.y, 3), "hp": u.health, "hp_max": u.health_max, "sh": u.shield,
                            "sh_max": u.shield_max} for u in obs.observation.raw_data.units if u.owner in (1, 2)])
            if obs.player_result or c.status != Status.in_replay:
                break
            try:
                c.step(a.step)
            except protocol.ProtocolError:
                break
    json.dump({"loops": loops, "units": frames}, open(a.out, "w"))
    print(f"wrote {a.out}: {len(loops)} observations, loops {loops[0]}..{loops[-1]}")


if __name__ == "__main__":
    main()
