"""Behaviour statistics from mappo.py episode dumps (eval_episodes.jsonl / episodes.jsonl).

usage: python scripts/behavior_stats.py FILE [FILE...]  -> one row per (file, iter, team)
"""
import json, re, sys, math
from collections import defaultdict, Counter

SELF = re.compile(r"You are agent a(\d), a (\w+) at \(([-\d.]+), ([-\d.]+)\), hp (\d+)/(\d+), shield (\d+)/(\d+)")
ENEMY = re.compile(r"^\s+e(\d+) (\w+) at \(([-\d.]+), ([-\d.]+)\), dist ([\d.]+), hp (\d+)/(\d+), shield (\d+)/(\d+)(.*)$", re.M)
DIRS = {"move_north": (0, 1), "move_south": (0, -1), "move_east": (1, 0), "move_west": (-1, 0)}


def parse(prompt):
    m = SELF.search(prompt)
    me = dict(x=float(m[3]), y=float(m[4]), hp=int(m[5]) + int(m[7]), mx=int(m[6]) + int(m[8]), typ=m[2])
    vis = prompt.split("Visible enemies:")[1].split("Units alive")[0] if "Visible enemies:" in prompt else ""
    en = [dict(id=f"e{g[0]}", typ=g[1], x=float(g[2]), y=float(g[3]), d=float(g[4]),
               hs=int(g[5]) + int(g[7]), inr="[in range]" in g[9]) for g in ENEMY.findall(vis)]
    return me, en


def action_of(reply):
    try:
        return json.loads(reply).get("action", ""), (json.loads(reply).get("msg") or "")
    except Exception:
        m = re.search(r'"action"\s*:\s*"([^"]+)"', reply or "")
        return (m[1] if m else ""), ""


def stats(eps):
    c = Counter()
    msgs = Counter()
    for ep in eps:
        c["ep"] += 1
        c["win"] += bool(ep["info"].get("battle_won"))
        c["steps"] += len(ep["rewards"])
        c["dead_allies"] += ep["info"].get("dead_allies", 0)
        c["dead_enemies"] += ep["info"].get("dead_enemies", 0)
        by_t = defaultdict(list)
        first = None
        for d in ep["decisions"]:
            me, en = parse(d["prompt"])
            act, msg = action_of(d["reply"])
            c["dec"] += 1
            c["tok"] += len(d["ids"])
            c["invalid"] += not d["valid"]
            kind = act.split("_")[0] if act else "none"
            c["k_" + kind] += 1
            c["msg_empty"] += not msg.strip()
            c["msg_words"] += len(msg.split())
            msgs[re.sub(r"[ea]\d", "X", msg.strip().lower())[:40]] += 1
            if en and first is None:
                first = d["t"]
            inr = [e for e in en if e["inr"]]
            if inr:
                c["inr"] += 1
                c["inr_attack"] += kind == "attack"
                if kind == "attack":
                    tgt = act.split("_", 1)[1]
                    c["att_inr"] += 1
                    c["att_weakest"] += tgt == min(inr, key=lambda e: e["hs"])["id"]
                    c["att_nearest"] += tgt == min(inr, key=lambda e: e["d"])["id"]
                    by_t[d["t"]].append(tgt)
            if en and act in DIRS:
                near = min(en, key=lambda e: e["d"])
                dx, dy = DIRS[act]
                toward = (near["x"] - me["x"]) * dx + (near["y"] - me["y"]) * dy > 0
                c["mv_vis"] += 1
                c["mv_away"] += not toward
                if me["hp"] < 0.3 * me["mx"]:
                    c["mv_low"] += 1
                    c["mv_low_away"] += not toward
            if en and me["hp"] < 0.3 * me["mx"]:
                c["low_vis"] += 1
                c["low_move"] += kind == "move"
        c["first_contact"] += first if first is not None else len(ep["rewards"])
        for t, ts in by_t.items():
            if len(ts) >= 2:
                c["focus_steps"] += 1
                c["focus_share"] += Counter(ts).most_common(1)[0][1] / len(ts)
    f = lambda a, b: c[a] / max(c[b], 1)
    row = dict(
        n=c["ep"], win=f("win", "ep"), steps=f("steps", "ep"), contact=f("first_contact", "ep"),
        dA=f("dead_allies", "ep"), dE=f("dead_enemies", "ep"),
        attack=f("k_attack", "dec"), move=f("k_move", "dec"), stop=f("k_stop", "dec"), invalid=f("invalid", "dec"),
        att_in_range=f("inr_attack", "inr"), weakest=f("att_weakest", "att_inr"), nearest=f("att_nearest", "att_inr"),
        focus=f("focus_share", "focus_steps"), move_away=f("mv_away", "mv_vis"),
        low_hp_move=f("low_move", "low_vis"), low_away=f("mv_low_away", "mv_low"),
        msg_empty=f("msg_empty", "dec"), msg_words=f("msg_words", "dec"), tokens=f("tok", "dec"))
    return row, msgs


def main():
    for path in sys.argv[1:]:
        groups = defaultdict(list)
        for l in open(path):
            ep = json.loads(l)
            groups[(ep["iter"], ep.get("team", "A"))].append(ep)
        for (it, team), eps in sorted(groups.items()):
            row, msgs = stats(eps)
            print(json.dumps(dict(file=path, iter=it, team=team, **{k: round(v, 3) for k, v in row.items()},
                                  top_msgs=msgs.most_common(6))))


if __name__ == "__main__":
    main()
