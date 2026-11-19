"""Build artifacts/battle_replay/demo3_video/index.html: the demo3 viewer with the real SC2 rendering as the battlefield.

Videos (a-d.mp4, next to index.html) come from scripts/render_demo3_video.sh: one frame per game loop at 16 fps,
so env step t (game loop OFFSET + 8 t, OFFSET = 2) is at video time (2 + 8 t) / 16 s and 1x = 2 steps/s.
Unit positions/hp/shields per game loop come from the replays (scripts/dump_replay_units.py); ids are mapped to
tags by build_demo3.build_from_replay (fit against the prompts). Labels are projected with the camera fitted by
scripts/calib_video_overlay.py.

usage: python scripts/build_demo3_video.py
"""
import base64
import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_battle_replay import README  # noqa: E402
from build_demo3 import build_from_replay, caption, find_ep  # noqa: E402

OUT = Path("artifacts/battle_replay/demo3_video")
# pinhole camera (f, D, pitch, ox, oy, target y) for --camera fixed --zoom 62 --size 1080x1080 on the 32x32 map,
# fitted by scripts/calib_video_overlay.py on all four videos (33 isolated-unit matches, median residual 1.4 px)
CAM = [1367.7596054980163, 31.65560701823288, 1.1954687408925466, 538.7697930882259, 1153.2569541875919, -2.907612198239324]
FPS = 16
SCEN = 100046
MAP = "external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map"


def play_log_as_episode(path):
    """smac_llm.play log (one episode) -> train.py-style episode dict for build_from_replay."""
    rows = [json.loads(l) for l in open(path)]
    summ = [r for r in rows if r.get("summary")][0]
    dec = [{"t": r["step"], "agent": str(d["agent"]), "prompt": d["prompt"], "reply": d["reply"], "valid": d["valid"]}
           for r in rows if not r.get("summary") for d in r["team_a"]]
    return {"decisions": dec, "rewards": [0.0] * summ["steps"],
            "info": {"battle_won": summ["won"], "dead_allies": summ["dead_allies"], "dead_enemies": summ["dead_enemies"]}}


def track(dump, maps):
    out = {}
    for side, key in (("al", "al"), ("en", "en")):
        out[side] = {}
        for i, tag in maps[key].items():
            seq, last = [], None
            for f in dump["units"]:
                u = next((u for u in f if u["tag"] == tag), None)
                if u is not None and u["hp"] > 0:
                    last = u
                    seq.append([round(u["x"], 2), round(u["y"], 2), round(u["hp"], 1), round(u["sh"], 1)])
                else:
                    seq.append([round(last["x"], 2), round(last["y"], 2), 0, 0] if last else [0, 0, 0, 0])
            out[side][str(i)] = seq
    return out


def main():
    mix = "colossus + 2 stalkers + 2 zealots on both sides"
    specs = [
        ("a", "Gemma 4 31B · C0 (no comm) · Loss", "Gemma 4 31B (ours)", False,
         play_log_as_episode("runs/demo3v/none/scen100046_11.jsonl"), "runs/demo3v/none_units.json",
         "runs/demo3v/none/replays/scen100046_11/scen100046_lost_s52_ep0.SC2Replay",
         f"C0 vs C2 comparison (this tab and the next): scenario {SCEN}, {mix}, identical start positions for both teams. "
         "Here the units cannot talk to each other."),
        ("b", "Gemma 4 31B · C2 · Win (with colossus)", "Gemma 4 31B (ours)", True,
         play_log_as_episode("runs/demo3v/struct/scen100046_27.jsonl"), "runs/demo3v/struct_units.json",
         "runs/demo3v/struct/replays/scen100046_27/scen100046_won_s44_ep0.SC2Replay",
         f"C0 vs C2 comparison (previous tab and this one): scenario {SCEN}, {mix}, identical start positions for both teams. "
         "Our colossus is a0."),
        ("c", "Gemma 4 E4B untrained · C2 · Loss", "Gemma 4 E4B (ours)", True,
         find_ep("runs/demo_untrained/episodes.jsonl", SCEN), "runs/demo_untrained/replay_units_scen100046.json",
         "runs/demo_untrained/replays/env2/scen100046_lost_s32_103743.SC2Replay",
         f"BEFORE training — scenario {SCEN}: same unit mix and same start positions for our squad as the next tab "
         "(after SFT on 31B games); the enemy start positions differ."),
        ("d", "Gemma 4 E4B after SFT on 31B games · C2 · Win", "Gemma 4 E4B (ours)", True,
         find_ep("runs/demo_awr/episodes.jsonl", SCEN), "runs/demo_awr/replay_units_scen100046.json",
         "runs/demo_awr/replays/env2/scen100046_won_s45_104443.SC2Replay",
         f"AFTER training — the same scenario {SCEN} as the previous tab (untrained model): same unit mix and our "
         "start positions; the enemy start positions differ."),
    ]
    games, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for k, label, name_a, chat, ep, dump_path, rep, note in specs:
            dump = json.load(open(dump_path))
            frames, summ, maps = build_from_replay(ep, dump, k, return_maps=True)
            loops = json.load(open(OUT / f"{k}.loops.json"))["loops"]
            assert loops[: len(loops)] == list(range(len(loops))), "video must have one frame per game loop from 0"
            assert len(loops) >= maps["offset"] + 8 * summ["steps"], (k, len(loops), summ["steps"])
            if not chat:
                for f in frames:
                    for o in f["a"].values():
                        o["say"] = None
            games.append({"label": label, "a": name_a, "b": "Built-in AI (enemy)", "chat": chat, "summary": summ,
                          "frames": frames, "final": True, "video": f"{k}.mp4", "fps": FPS, "offset": maps["offset"],
                          "cam": CAM, "track": track(dump, maps),
                          "caption": caption(frames, summ, note + " Video: the real StarCraft II rendering of this game's replay.")})
            z.write(rep, f"tab{len(games)}_{k}_{label.split(' · ')[0].replace(' ', '')}_{summ['winner']}.SC2Replay")
        z.write(MAP, "32x32_flat.SC2Map")
        z.write(MAP.replace(".SC2Map", "_2p.SC2Map"), "32x32_flat_2p.SC2Map")
        z.writestr("README.txt", README)
    data = {"title": "Gemma Squads · C0 vs C2 Team Chat", "channel": "c0/c2", "games": games,
            "zip": base64.b64encode(buf.getvalue()).decode(), "zip_kb": round(len(buf.getvalue()) / 1024), "n_replays": len(games)}
    html = Path("scripts/battle_template_video.html").read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (OUT / "index.html").write_text(html)
    print(f"wrote {OUT / 'index.html'} ({len(html) // 1024} KB)")
    for g in games:
        print("  ", g["label"], "|", g["caption"])


if __name__ == "__main__":
    main()
