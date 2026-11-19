"""Build the game-style battle viewer (artifacts/battle_replay/index.html) from logged games,
with the real .SC2Replay files and the SMAC map bundled as a downloadable zip.

usage: python scripts/build_battle_replay.py --out artifacts/battle_replay/index.html \
           --game "runs/replays_sc2/games.jsonl|Gemma 4 31B|Built-in AI|runs/replays_sc2" \
           [--game "runs/versus_or/gemma_vs_qwen.jsonl|Gemma 4 31B|Qwen3 235B|runs/versus_or/replays_gq" ...]
"""
import argparse
import base64
import glob
import io
import json
import os
import re
import zipfile
from pathlib import Path

MOVES = {0: "no_op", 1: "stop", 2: "move_north", 3: "move_south", 4: "move_east", 5: "move_west"}
N_NO_ATTACK = 6

README = """LLM-controlled 5-unit Protoss squads in SMACv2 (StarCraft II 4.10, Linux build).
Each unit is a separate model call. Opponents: the game's built-in AI (difficulty 7) or another model
(model-vs-model games use 32x32_flat_2p.SC2Map, a two-player copy of the map).

How to watch these replays in StarCraft II
1. Install StarCraft II (free to play) from Battle.net and start it once.
2. Put 32x32_flat.SC2Map and 32x32_flat_2p.SC2Map in <StarCraft II folder>/Maps/SMAC_Maps/ (create the folders).
   Windows: C:\\Program Files (x86)\\StarCraft II\\Maps\\SMAC_Maps\\
   macOS:   /Applications/StarCraft II/Maps/SMAC_Maps/
3. Open a .SC2Replay file (double-click, or Replays -> Browse in the game).
   The replays were recorded on game version 4.10 (build 75689); if the client asks,
   let it download that version.

Files
  game<N>_<teamA>_vs_<teamB>_<winner>.SC2Replay   one file per game (winner A, B or draw)
  32x32_flat.SC2Map / 32x32_flat_2p.SC2Map        the SMACv2 maps (units are spawned by the environment)
"""


def act_name(a):
    return MOVES.get(a, f"attack_e{a - N_NO_ATTACK}")


def _order(r):
    m = re.search(r"\{.*?\}", r["reply"] or "", re.S)
    return {"act": act_name(r["action"]), "valid": r["valid"], "reply": (m.group(0) if m else r["reply"] or "")[:300],
            "obs": r["prompt"], "lat": r.get("latency"), "say": r.get("say") or r.get("msg")}


def load_games(path):
    """Episodes from smac_llm.run (vs built-in AI) or smac_llm.versus (model vs model) logs."""
    eps = {}
    for line in open(path):
        d = json.loads(line)
        e = eps.setdefault(d["episode"], {"frames": [], "summary": None})
        if d.get("summary"):
            e["summary"] = d
            continue
        recs_a = d.get("agents", d.get("team_a", []))
        recs_b = d.get("team_b", [])
        e["frames"].append({"t": d["step"], "u": d["units"],
                            "a": {str(r["agent"]): _order(r) for r in recs_a},
                            "b": {str(r["agent"]): _order(r) for r in recs_b}})
    out = []
    for k in sorted(eps):
        s = eps[k]["summary"]
        if "winner" not in s:  # run.py summary -> common form
            s = {**s, "winner": "A" if s["won"] else ("B" if s.get("dead_allies") == 5 else "draw")}
        out.append({"frames": eps[k]["frames"], "summary": s})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--game", action="append", required=True,
                   help="LOG|TEAM_A_NAME|TEAM_B_NAME|REPLAY_DIR (repeatable)")
    p.add_argument("--map", default="external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map")
    p.add_argument("--title", default="Gemma Squad Battle")
    p.add_argument("--channel", default="none")
    p.add_argument("--template", default="scripts/battle_template.html")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out_games, buf = [], io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for spec in a.game:
            log, name_a, name_b, rdir = spec.split("|")
            reps = sorted(glob.glob(os.path.join(rdir, "*.SC2Replay")))
            for i, g in enumerate(load_games(log)):
                n += 1
                s = g["summary"]
                w = s["winner"]
                res = f"{name_a} wins" if w == "A" else f"{name_b} wins" if w == "B" else "Draw (time limit)"
                chat = any(o.get("say") for f in g["frames"] for t in ("a", "b") for o in f[t].values())
                out_games.append({"label": f"Game {n} · {res}", "a": name_a, "b": name_b, "chat": chat,
                                  "summary": s, "frames": g["frames"]})
                if i < len(reps):
                    slug = lambda x: re.sub(r"[^A-Za-z0-9]+", "", x)
                    z.write(reps[i], f"game{n}_{slug(name_a)}_vs_{slug(name_b)}_{w}.SC2Replay")
        z.write(a.map, "32x32_flat.SC2Map")
        z.write(a.map.replace(".SC2Map", "_2p.SC2Map"), "32x32_flat_2p.SC2Map")
        z.writestr("README.txt", README)
    data = {"title": a.title, "channel": a.channel, "games": out_games,
            "zip": base64.b64encode(buf.getvalue()).decode(), "zip_kb": round(len(buf.getvalue()) / 1024)}
    html = Path(a.template).read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html) // 1024} KB), games={[g['label'] for g in out_games]}, zip={data['zip_kb']} KB")


if __name__ == "__main__":
    main()
