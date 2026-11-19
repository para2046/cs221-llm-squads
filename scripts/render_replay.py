"""Render a .SC2Replay with the real StarCraft II renderer (headless Linux build, EGL/OSMesa) to an mp4.

usage: SC2PATH=external/sc2/StarCraftII .venv-smac/bin/python scripts/render_replay.py REPLAY.SC2Replay OUT.mp4
         [--map external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map] [--size 1280x720] [--step-mul 4]
         [--fps 24] [--max-frames N] [--zoom 18]
The camera follows the centre of all living units each frame.
"""
import argparse

import os
import sys

import imageio
from absl import flags
import numpy as np
from pysc2 import run_configs
from pysc2.lib import protocol
from pysc2.lib.remote_controller import Status
from s2clientprotocol import common_pb2, sc2api_pb2 as sc_pb, spatial_pb2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("replay")
    p.add_argument("out")
    p.add_argument("--map", default="external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map")
    p.add_argument("--size", default="1280x720")
    p.add_argument("--step-mul", type=int, default=4)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--max-frames", type=int, default=100000)
    p.add_argument("--zoom", type=float, default=18.0, help="camera width in world units")
    p.add_argument("--player", type=int, default=1)
    p.add_argument("--camera", choices=["auto", "fixed", "follow"], default="auto",
                   help="auto: frame all living units (zoom to their spread, kept inside the map, smoothed); "
                        "fixed: map centre at --zoom; follow: unit centroid at --zoom")
    a = p.parse_args()
    w, h = map(int, a.size.split("x"))

    flags.FLAGS([sys.argv[0]])
    rc = run_configs.get()
    replay_data = open(os.path.abspath(a.replay), "rb").read()
    opts = sc_pb.InterfaceOptions(
        raw=True, score=False,
        render=sc_pb.SpatialCameraSetup(resolution=common_pb2.Size2DI(x=w, y=h),
                                        minimap_resolution=common_pb2.Size2DI(x=128, y=128),
                                        width=a.zoom))
    with rc.start(want_rgb=True, full_screen=False) as c:
        c.start_replay(sc_pb.RequestStartReplay(replay_data=replay_data, map_data=open(a.map, "rb").read(),
                                                options=opts, observed_player_id=a.player, disable_fog=True))
        gi = c.game_info()
        pa = gi.start_raw.playable_area
        mx, my = (pa.p0.x + pa.p1.x) / 2, (pa.p0.y + pa.p1.y) / 2
        print(f"map {gi.start_raw.map_size.x}x{gi.start_raw.map_size.y}, playable ({pa.p0.x},{pa.p0.y})-({pa.p1.x},{pa.p1.y})")
        writer = imageio.get_writer(a.out, fps=a.fps, codec="libx264", quality=8, macro_block_size=8)
        n = 0
        while n < a.max_frames and c.status == Status.in_replay:
            obs = c.observe()
            units = [u for u in obs.observation.raw_data.units if u.alliance in (1, 4) and u.health > 0
                     and u.owner in (1, 2)]
            if a.camera == "auto" and units and c.status == Status.in_replay:
                xs, ys = [u.pos.x for u in units], [u.pos.y for u in units]
                ext = max(max(xs) - min(xs), max(ys) - min(ys)) + 6.0           # spread + margin
                d = min(max(2.0 * ext, 28.0), 64.0)                            # visible width ~ 0.5 * distance
                half = 0.25 * d
                tx = min(max((max(xs) + min(xs)) / 2, pa.p0.x + half - 2), pa.p1.x - half + 2)
                ty = min(max((max(ys) + min(ys)) / 2, pa.p0.y + half - 2), pa.p1.y - half + 2)
                cam = getattr(main, "cam", None) or [tx, ty, d]
                cam = [cam[0] + 0.15 * (tx - cam[0]), cam[1] + 0.15 * (ty - cam[1]), cam[2] + 0.15 * (d - cam[2])]
                main.cam = cam
                c.observer_act(sc_pb.ObserverAction(camera_move=sc_pb.ActionObserverCameraMove(
                    world_pos=common_pb2.Point2D(x=cam[0], y=cam[1]), distance=cam[2])))
            elif a.camera == "fixed" and c.status == Status.in_replay:
                c.observer_act(sc_pb.ObserverAction(camera_move=sc_pb.ActionObserverCameraMove(
                    world_pos=common_pb2.Point2D(x=mx, y=my), distance=a.zoom)))
            elif a.camera == "follow" and units and c.status == Status.in_replay:
                x = float(np.mean([u.pos.x for u in units]))
                y = float(np.mean([u.pos.y for u in units]))
                c.observer_act(sc_pb.ObserverAction(camera_move=sc_pb.ActionObserverCameraMove(
                    world_pos=common_pb2.Point2D(x=x, y=y), distance=a.zoom)))
            img = obs.observation.render_data.map
            if img.data:
                arr = np.frombuffer(img.data, dtype=np.uint8).reshape(img.size.y, img.size.x, img.bits_per_pixel // 8)
                writer.append_data(arr[:, :, :3])
                n += 1
            if obs.player_result or c.status != Status.in_replay:
                break
            try:
                c.step(a.step_mul)
            except protocol.ProtocolError:
                break
        writer.close()
    print(f"wrote {a.out}: {n} frames")


if __name__ == "__main__":
    main()
