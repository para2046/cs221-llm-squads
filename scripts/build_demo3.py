"""Build artifacts/battle_replay/demo3.html: three games in the battle viewer (scripts/battle_template_demo3.html).

  A. Gemma 4 31B, C2 (smac_llm.play log: has per-step unit snapshots)
  B/C. Gemma 4 E4B untrained vs after AWR training, same scenario (train.py episode logs: no unit snapshots).
       Unit positions / hp / shields for B and C come from the real .SC2Replay files, dumped with
       scripts/dump_replay_units.py (every game loop). Env step t is matched to game loop OFFSET + 8 t
       (SMAC step_mul 8); OFFSET and the agent/enemy id <-> unit tag maps are fitted to the exact
       positions / hp / shields written in the prompts, and the fit error is printed.

usage: python scripts/build_demo3.py --dump-untrained U.json --dump-awr A.json [--out artifacts/battle_replay/demo3.html]
"""
import argparse
import base64
import io
import itertools
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_battle_replay import README, load_games  # noqa: E402

TYPES = {1971: "colossus", 1976: "stalker", 1977: "zealot", 4: "colossus", 74: "stalker", 73: "zealot"}
STEP_MUL = 8
SELF = re.compile(r"You are agent a(\d), an? (\w+) at \(([-\d.]+), ([-\d.]+)\), hp ([\d.]+)/[\d.]+, shield ([\d.]+)/")
ENEMY = re.compile(r"^\s+e(\d) (\w+) at \(([-\d.]+), ([-\d.]+)\), dist [\d.]+, hp ([\d.]+)/[\d.]+, shield ([\d.]+)/", re.M)


def visible_enemies(prompt):
    sec = prompt.split("Visible enemies:")[1].split("Units alive")[0] if "Visible enemies:" in prompt else ""
    return [int(m.group(1)) for m in ENEMY.finditer(sec)]


def enemy_sightings(prompt):
    """(id, type, x, y, hp, sh) of enemies in the unit's own view."""
    sec = prompt.split("Visible enemies:")[1].split("Units alive")[0] if "Visible enemies:" in prompt else ""
    return [(int(m[1]), m[2], float(m[3]), float(m[4]), float(m[5]), float(m[6])) for m in ENEMY.finditer(sec)]


def parse_reply(reply):
    m = re.search(r"\{.*?\}", reply or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    return d, (m.group(0) if m else reply or "")


def match(costs):
    """min-cost assignment of ids 0..n-1 to columns (n <= 6, brute force). costs[id][col]."""
    n, cols = len(costs), range(len(costs[0]))
    best = min(itertools.permutations(cols, n), key=lambda p: sum(costs[i][p[i]] for i in range(n)))
    return list(best)


def state_at(dump, loop, tag):
    """unit dict at a game loop, or None if gone (dead)."""
    k = min(loop, len(dump["units"]) - 1)
    for u in dump["units"][k]:
        if u["tag"] == tag:
            return u
    return None


def build_from_replay(ep, dump, name, return_maps=False):
    dec = ep["decisions"]
    steps = len(ep["rewards"])
    by_t = {}
    for d in dec:
        by_t.setdefault(d["t"], {})[int(d["agent"])] = d
    selfs = {(t, a): SELF.search(d["prompt"]) for t, ds in by_t.items() for a, d in ds.items()}
    tags = {o: sorted({u["tag"] for f in dump["units"] for u in f if u["owner"] == o}) for o in (1, 2)}
    ttype = {u["tag"]: TYPES.get(u["type"], str(u["type"])) for f in dump["units"] for u in f}

    def err(u, x, y, hp, sh):
        return 50.0 if u is None else abs(u["x"] - x) + abs(u["y"] - y) + abs(u["hp"] - hp) / 10 + abs(u["sh"] - sh) / 10

    # 1) offset + ally map: fit the agents' self-reports (exact own pos/hp/shield each step)
    fits = {}
    for off in range(0, 12):
        costs = [[sum(err(state_at(dump, off + STEP_MUL * t, tg), float(m[3]), float(m[4]), float(m[5]), float(m[6]))
                      + (100 if ttype[tg] != m[2] else 0) for (t, a), m in selfs.items() if a == aid)
                  for tg in tags[1]] for aid in range(5)]
        perm = match(costs)
        fits[off] = (sum(costs[i][perm[i]] for i in range(5)), perm)
    off = min(fits, key=lambda k: fits[k][0])
    ally_tag = {aid: tags[1][fits[off][1][aid]] for aid in range(5)}
    print(f"[{name}] offset fit (total self-report error by game-loop offset): "
          + ", ".join(f"{k}:{v[0]:.1f}" for k, v in fits.items()) + f" -> offset {off}")
    # 2) enemy map: fit every enemy sighting in the prompts
    sights = [(t, s) for t, ds in by_t.items() for d in ds.values() for s in enemy_sightings(d["prompt"])]
    costs = [[sum(err(state_at(dump, off + STEP_MUL * t, tg), s[2], s[3], s[4], s[5]) + (100 if ttype[tg] != s[1] else 0)
                  for t, s in sights if s[0] == eid) for tg in tags[2]] for eid in range(5)]
    perm = match(costs)
    enemy_tag = {eid: tags[2][perm[eid]] for eid in range(5)}
    # alignment report: max abs error of self-reports and sightings at the fitted offset
    mx = [0, 0, 0]
    for (t, a), m in selfs.items():
        u = state_at(dump, off + STEP_MUL * t, ally_tag[a])
        mx = [max(mx[0], abs(u["x"] - float(m[3])), abs(u["y"] - float(m[4]))), max(mx[1], abs(u["hp"] - float(m[5]))),
              max(mx[2], abs(u["sh"] - float(m[6])))]
    me = [0, 0, 0]
    for t, s in sights:
        u = state_at(dump, off + STEP_MUL * t, enemy_tag[s[0]])
        me = [max(me[0], abs(u["x"] - s[2]), abs(u["y"] - s[3])), max(me[1], abs(u["hp"] - s[4])), max(me[2], abs(u["sh"] - s[5]))]
    print(f"[{name}] {len(selfs)} self-reports: max |pos| {mx[0]:.3f}, |hp| {mx[1]:.1f}, |shield| {mx[2]:.1f}; "
          f"{len(sights)} enemy sightings: max |pos| {me[0]:.3f}, |hp| {me[1]:.1f}, |shield| {me[2]:.1f}")

    frames, last = [], {}
    for t in range(steps + 1):
        loop = off + STEP_MUL * t
        u_out = {"allies": [], "enemies": []}
        for side, tmap in (("allies", ally_tag), ("enemies", enemy_tag)):
            for i, tg in tmap.items():
                u = state_at(dump, loop, tg)
                if u is not None and u["hp"] > 0:
                    last[(side, i)] = u
                    rec = {"id": i, "type": ttype[tg], "x": round(u["x"], 2), "y": round(u["y"], 2), "hp": u["hp"],
                           "hp_max": u["hp_max"], "sh": u["sh"], "sh_max": u["sh_max"]}
                else:  # destroyed: keep it at its last position with hp 0
                    p = last.get((side, i)) or {"x": 0, "y": 0, "hp_max": 1, "sh_max": 0}
                    rec = {"id": i, "type": ttype[tg], "x": round(p["x"], 2), "y": round(p["y"], 2), "hp": 0.0,
                           "hp_max": p["hp_max"], "sh": 0.0, "sh_max": p["sh_max"]}
                u_out[side].append(rec)
        orders, vis = {}, set()
        for a, d in sorted(by_t.get(t, {}).items()):
            js, raw = parse_reply(d["reply"])
            orders[str(a)] = {"act": str(js.get("action") or "no_op"), "valid": d["valid"], "reply": raw[:300],
                              "obs": d["prompt"], "lat": None, "say": (js.get("msg") or None)}
            vis.update(visible_enemies(d["prompt"]))
        if t == steps and frames:  # final frame (no prompts): keep the last step's view
            vis = set(frames[-1]["vis"])
        frames.append({"t": t, "u": u_out, "a": orders, "b": {}, "vis": sorted(vis)})
    info = ep["info"]
    won = bool(info["battle_won"])
    summ = {"winner": "A" if won else ("B" if info["dead_allies"] == 5 else "draw"), "won": won, "steps": steps,
            "dead_allies": info["dead_allies"], "dead_enemies": info["dead_enemies"],
            "invalid_rate": sum(not d["valid"] for d in dec) / max(len(dec), 1)}
    # cross-check the end state against the env's own counts
    end = frames[-1]["u"]
    da, de = sum(u["hp"] <= 0 for u in end["allies"]), sum(u["hp"] <= 0 for u in end["enemies"])
    print(f"[{name}] replay end state: allies dead {da}, enemies dead {de}; env info: {info['dead_allies']}, {info['dead_enemies']}")
    return (frames, summ, {"offset": off, "al": ally_tag, "en": enemy_tag}) if return_maps else (frames, summ)


def caption(frames, summ, extra=""):
    ords = [o for f in frames for o in f["a"].values()]
    pct = 100 * sum(bool(o.get("say")) for o in ords) / max(len(ords), 1)
    res = "Win" if summ["winner"] == "A" else "Loss" if summ["winner"] == "B" else "Draw"
    return (f"{res} in {summ['steps']} steps · our units lost {summ['dead_allies']}/5 · enemies killed "
            f"{summ['dead_enemies']}/5 · "
            + (f"{pct:.0f}% of {len(ords)} unit decisions carried a team message. " if pct else f"{len(ords)} unit decisions, no team messages. ")
            + extra).strip()


def find_ep(path, scen):
    for line in open(path):
        d = json.loads(line)
        if d["scenario"] == scen:
            return d
    raise SystemExit(f"scenario {scen} not in {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump-untrained", required=True)
    p.add_argument("--dump-awr", required=True)
    p.add_argument("--scenario", type=int, default=100046)
    p.add_argument("--out", default="artifacts/battle_replay/demo3.html")
    p.add_argument("--template", default="scripts/battle_template_demo3.html")
    p.add_argument("--map", default="external/sc2/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map")
    a = p.parse_args()
    enemy = "Built-in AI (enemy)"
    games, reps = [], []

    # 31B, E1 cells: C0 loss and C2 win with the same unit mix on both sides (colossus + 2 stalkers + 2 zealots);
    # the cells were not run on identical scenarios, so start positions differ. No .SC2Replay exists for these.
    mix = "colossus + 2 stalkers + 2 zealots on both sides"
    for log, ep_i, lab, chat, note in (
            ("runs/e1u_31b_none/w2.jsonl", 1, "Gemma 4 31B · C0 (no comm) · Loss", False,
             f"C0 vs C2 comparison (this tab and the next): same unit mix, {mix}, but not the identical start positions. "
             "Here the units cannot talk to each other."),
            ("runs/e1u_31b_struct/w8.jsonl", 1, "Gemma 4 31B · C2 · Win (with colossus)", True,
             f"C0 vs C2 comparison (previous tab and this one): same unit mix, {mix}, but not the identical start positions. "
             "Our colossus is a0.")):
        eps = sorted({json.loads(l)["episode"] for l in open(log)})
        g = load_games(log)[eps.index(ep_i)]
        raw = [r for r in map(json.loads, open(log)) if r["episode"] == ep_i and not r.get("summary")]
        for f, r in zip(g["frames"], raw):
            f["vis"] = sorted({e for d in r["team_a"] for e in visible_enemies(d["prompt"])})
        s = g["summary"]
        games.append({"label": lab, "a": "Gemma 4 31B (ours)", "b": enemy, "chat": chat, "summary": s, "frames": g["frames"],
                      "caption": caption(g["frames"], s, note + f" Source: {log} episode {ep_i}; map positions are the env's own per-step unit snapshots.")})

    scen = a.scenario
    for key, path, dump, rep, lab, note in (
            ("untrained", "runs/demo_untrained/episodes.jsonl", a.dump_untrained,
             "runs/demo_untrained/replays/env2/scen100046_lost_s32_103743.SC2Replay", "Gemma 4 E4B untrained",
             f"BEFORE training — scenario {scen}: same unit mix and same start positions for our squad as the next tab (after SFT on 31B games); the enemy start positions differ."),
            ("awr", "runs/demo_awr/episodes.jsonl", a.dump_awr,
             "runs/demo_awr/replays/env2/scen100046_won_s45_104443.SC2Replay", "Gemma 4 E4B after SFT on 31B games",
             f"AFTER training — the same scenario {scen} as the previous tab (untrained model): same unit mix and our start positions; the enemy start positions differ.")):
        ep = find_ep(path, scen)
        frames, summ = build_from_replay(ep, json.load(open(dump)), key)
        res = "Win" if summ["winner"] == "A" else "Loss" if summ["winner"] == "B" else "Draw"
        games.append({"label": f"{lab} · C2 · {res}", "a": "Gemma 4 E4B (ours)", "b": enemy, "chat": True,
                      "summary": summ, "frames": frames, "final": True,
                      "caption": caption(frames, summ, note + " Map positions come from the game's .SC2Replay.")})
        reps.append((rep, f"game{len(games)}_Gemma4_E4B_{key}_{summ['winner']}"))

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for rp, nm in reps:
            z.write(rp, f"{nm}.SC2Replay")
        z.write(a.map, "32x32_flat.SC2Map")
        z.write(a.map.replace(".SC2Map", "_2p.SC2Map"), "32x32_flat_2p.SC2Map")
        z.writestr("README.txt", README)
    data = {"title": "Gemma Squads · C0 vs C2 Team Chat", "channel": "c0/c2", "games": games,
            "zip": base64.b64encode(buf.getvalue()).decode(), "zip_kb": round(len(buf.getvalue()) / 1024), "n_replays": len(reps)}
    html = Path(a.template).read_text().replace("/*__DATA__*/null", json.dumps(data))
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({len(html) // 1024} KB), tabs={[g['label'] for g in games]}")
    for gm in games:
        print("  ", gm["caption"])


if __name__ == "__main__":
    main()
