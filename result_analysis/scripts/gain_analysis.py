"""Where does a win-rate gain come from? Per-game and per-decision features of several runs on the same maps.

usage: python scripts/gain_analysis.py NAME=runs/x/eval_episodes.jsonl NAME2=... [--json out.json]
Every feature is read from the logged prompts (what the units saw) and replies; nothing is re-simulated.
Groups: outcome / timing, engagement (formation), per-decision execution, kiting, messages.
"""
import json, re, sys, math
from collections import defaultdict, Counter

SELF = re.compile(r"You are agent a(\d), a (\w+) at \(([-\d.]+), ([-\d.]+)\), hp (\d+)/(\d+), shield (\d+)/(\d+), weapon cooldown (\d+)")
ENEMY = re.compile(r"^\s+e(\d+) (\w+) at \(([-\d.]+), ([-\d.]+)\), dist ([\d.]+), hp (\d+)/(\d+), shield (\d+)/(\d+)(.*)$", re.M)
MATE = re.compile(r"^\s+a(\d) (\w+) at \(([-\d.]+), ([-\d.]+)\), dist ([\d.]+)", re.M)
ALIVE = re.compile(r"Units alive: your team (\d+), enemy (\d+)")
DIRS = {"move_north": (0, 1), "move_south": (0, -1), "move_east": (1, 0), "move_west": (-1, 0)}


def parse(p):
    m = SELF.search(p)
    me = dict(a=int(m[1]), typ=m[2], x=float(m[3]), y=float(m[4]), hs=int(m[5]) + int(m[7]), mx=int(m[6]) + int(m[8]),
              cd=int(m[9]))
    vis = p.split("Visible enemies:")[1].split("Units alive")[0] if "Visible enemies:" in p else ""
    en = [dict(id=f"e{g[0]}", typ=g[1], x=float(g[2]), y=float(g[3]), d=float(g[4]), hs=int(g[5]) + int(g[7]),
               inr="[in range]" in g[9]) for g in ENEMY.findall(vis)]
    tm = p.split("Visible teammates:")[1].split("Visible enemies")[0] if "Visible teammates:" in p else ""
    mates = [dict(d=float(g[4])) for g in MATE.findall(tm)]
    al = ALIVE.search(p)
    return me, en, mates, (int(al[1]), int(al[2])) if al else None


def act_msg(reply):
    try:
        j = json.loads(reply)
        return j.get("action", "") or "", (j.get("msg") or "").strip()
    except Exception:
        m = re.search(r'"action"\s*:\s*"([^"]+)"', reply or "")
        return (m[1] if m else ""), ""


def game_features(ep):
    """One dict per game."""
    by_t = defaultdict(list)
    for d in ep["decisions"]:
        by_t[int(d["t"])].append(d)
    T = max(by_t) + 1 if by_t else 0
    alive = {}
    pos = {}
    for t, ds in by_t.items():
        for d in ds:
            me, en, mates, al = parse(d["prompt"])
            if al:
                alive[t] = al
            pos.setdefault(t, []).append((me["x"], me["y"], bool(en)))
    won = bool(ep["info"].get("battle_won"))
    dA, dE = ep["info"].get("dead_allies", 0), ep["info"].get("dead_enemies", 0)
    ts = sorted(alive)
    first_death = next((t for t in ts if alive[t][0] < 5), None)
    first_kill = next((t for t in ts if alive[t][1] < 5), None)
    contact = next((t for t in sorted(pos) if any(v for _, _, v in pos[t])), None)
    f = dict(won=won, draw=not won and dA < 5, steps=T, dA=dA, dE=dE, contact=contact,
             first_death=first_death, first_kill=first_kill)
    # who lost a unit first (ties: same step -> neither)
    fd = first_death if first_death is not None else 10**6
    fk = first_kill if first_kill is not None else 10**6
    f["kill_first"] = fk < fd
    f["die_first"] = fd < fk
    # formation at first contact: mean distance of our live units to their centroid; share engaged within 3 steps
    if contact is not None:
        P = pos[contact]
        cx, cy = sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P)
        f["spread_contact"] = sum(math.hypot(p[0] - cx, p[1] - cy) for p in P) / len(P)
        seen = set()
        for t in range(contact, contact + 4):
            for d in by_t.get(t, []):
                me, en, _, _ = parse(d["prompt"])
                if en:
                    seen.add(me["a"])
        f["engaged_3"] = len(seen) / max(len(P), 1)
    return f


def phase_of(al):
    """even / behind / ahead in alive units (from this unit's prompt)."""
    if al is None or al[0] == al[1]:
        return "even"
    return "behind" if al[0] < al[1] else "ahead"


def decision_counts(ep, phase=None):
    c = Counter()
    for _ in [0]:
        by_t = defaultdict(list)
        named = {}   # t -> {enemy ids named in messages at t}
        for d in ep["decisions"]:
            t = int(d["t"])
            me, en, mates, al = parse(d["prompt"])
            act, msg = act_msg(d["reply"])
            kind = act.split("_")[0] if act else "none"
            if phase and phase_of(al) != phase:
                if msg:
                    named.setdefault(t, set()).update(set(re.findall(r"\be\d\b", msg.lower())))
                continue
            c["dec"] += 1
            c["invalid"] += not d.get("valid", True)
            c["msg"] += bool(msg)
            if msg:
                ids = set(re.findall(r"\be\d\b", msg.lower()))
                c["msg_named"] += bool(ids)
                named.setdefault(t, set()).update(ids)
            if en:
                c["vis"] += 1
                c["vis_alone"] += not mates          # sees an enemy, no teammate in sight
            inr = [e for e in en if e["inr"]]
            if inr:
                c["inr"] += 1
                c["inr_attack"] += kind == "attack"
                if me["cd"] == 0:
                    c["ready"] += 1
                    c["ready_noattack"] += kind != "attack"   # weapon ready, target in range, no shot
                if kind == "attack":
                    tgt = act.split("_", 1)[1]
                    c["att"] += 1
                    c["att_weakest"] += tgt == min(inr, key=lambda e: e["hs"])["id"]
                    by_t[t].append(tgt)
            if en and kind == "move" and act in DIRS:
                near = min(en, key=lambda e: e["d"])
                dx, dy = DIRS[act]
                away = (near["x"] - me["x"]) * dx + (near["y"] - me["y"]) * dy <= 0
                c["mv_vis"] += 1
                c["mv_away"] += away
                if me["hs"] < 0.3 * me["mx"]:
                    c["low_mv"] += 1
                    c["low_mv_away"] += away
            # kiting: stalker reloading with a melee zealot within 2.5
            if me["typ"] == "stalker" and me["cd"] > 0 and any(e["typ"] == "zealot" and e["d"] < 2.5 for e in en):
                c["kite_opp"] += 1
                if kind == "move" and act in DIRS:
                    z = min((e for e in en if e["typ"] == "zealot"), key=lambda e: e["d"])
                    dx, dy = DIRS[act]
                    c["kite"] += (z["x"] - me["x"]) * dx + (z["y"] - me["y"]) * dy <= 0
            # follow: a named enemy (by a teammate at t-1) is in my range now -> do I attack it?
            prev = named.get(t - 1, set())
            cand = [e for e in inr if e["id"] in prev]
            if cand and len(inr) >= 2:
                c["fol_opp"] += 1
                c["fol"] += kind == "attack" and act.split("_", 1)[1] in {e["id"] for e in cand}
                c["fol_chance"] += len(cand) / len(inr)
        for t, ts in by_t.items():
            if len(ts) >= 2:
                c["focus_steps"] += 1
                c["focus_all"] += len(set(ts)) == 1
    return c


def decision_features(eps, phase=None):
    c = Counter()
    for ep in eps:
        c.update(decision_counts(ep, phase))
    r = lambda a, b: c[a] / max(c[b], 1)
    return dict(invalid=r("invalid", "dec"), msg=r("msg", "dec"), msg_named=r("msg_named", "msg"),
                vis_alone=r("vis_alone", "vis"), attack_in_range=r("inr_attack", "inr"),
                ready_noattack=r("ready_noattack", "ready"), weakest=r("att_weakest", "att"),
                focus_all=r("focus_all", "focus_steps"), move_away=r("mv_away", "mv_vis"),
                low_hp_away=r("low_mv_away", "low_mv"), kite=r("kite", "kite_opp"), kite_n=c["kite_opp"],
                follow=r("fol", "fol_opp"), follow_chance=r("fol_chance", "fol_opp"), follow_n=c["fol_opp"])


def load(path):
    eps = {}
    for l in open(path):
        e = json.loads(l)
        if e.get("team", "A") == "A" and e.get("scenario") is not None:
            eps.setdefault(e["scenario"], e)
    return eps


def summarize(eps, maps):
    G = [game_features(eps[k]) for k in maps]
    n = len(G)
    m = lambda key, sub=None: (lambda xs: sum(xs) / len(xs) if xs else float("nan"))(
        [g[key] for g in (sub or G) if g.get(key) is not None])
    lost1 = [g for g in G if g["die_first"]]
    kill1 = [g for g in G if g["kill_first"]]
    out = dict(n=n, win=m("won"), draw=m("draw"), dE=m("dE"), dA=m("dA"), steps=m("steps"), contact=m("contact"),
               first_death=m("first_death"), kill_first=m("kill_first"), die_first=m("die_first"),
               win_if_die_first=m("won", lost1), n_die_first=len(lost1),
               win_if_kill_first=m("won", kill1), n_kill_first=len(kill1),
               spread_contact=m("spread_contact"), engaged_3=m("engaged_3"))
    out.update(decision_features([eps[k] for k in maps]))
    return out


PHASE_KEYS = ["dec", "attack_in_range", "weakest", "focus_all", "move_away", "low_hp_away", "vis_alone", "follow",
              "follow_chance", "msg"]


def boot_diff(eps_a, eps_b, maps, key, B=2000, seed=0):
    """paired bootstrap over maps of (b - a) for a game-level 0/1 or numeric feature."""
    import random
    rnd = random.Random(seed)
    da = {k: game_features(eps_a[k]) for k in maps}
    db = {k: game_features(eps_b[k]) for k in maps}
    diffs = []
    for _ in range(B):
        s = [rnd.choice(maps) for _ in maps]
        diffs.append(sum(float(db[k][key] or 0) - float(da[k][key] or 0) for k in s) / len(s))
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


def boot_ratio(eps_a, eps_b, maps, num, den, phase=None, B=1000, seed=0):
    """paired bootstrap over maps of the difference in a decision-level ratio num/den."""
    import random
    rnd = random.Random(seed)
    ca = {k: decision_counts(eps_a[k], phase) for k in maps}
    cb = {k: decision_counts(eps_b[k], phase) for k in maps}
    diffs = []
    for _ in range(B):
        s = [rnd.choice(maps) for _ in maps]
        A, Bc = Counter(), Counter()
        for k in s:
            A.update(ca[k]); Bc.update(cb[k])
        diffs.append(Bc[num] / max(Bc[den], 1) - A[num] / max(A[den], 1))
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B)]


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if "=" in a]
    runs = {a.split("=", 1)[0]: load(a.split("=", 1)[1]) for a in args}
    maps = sorted(set.intersection(*(set(v) for v in runs.values())))
    res = {name: summarize(eps, maps) for name, eps in runs.items()}
    names = list(res)
    fmt = lambda v: f"{v:14.3f}" if isinstance(v, float) else f"{v:14d}"
    print(f"{len(maps)} shared maps")
    print(f"{'':20s}" + "".join(f"{n:>14s}" for n in names))
    for k in res[names[0]]:
        print(f"{k:20s}" + "".join(fmt(res[n][k]) for n in names))
    out = dict(maps=maps, res=res, phase={})
    for ph in ["even", "behind", "ahead"]:
        print(f"\n-- phase: {ph} (alive units ours vs enemy, as seen in each prompt)")
        out["phase"][ph] = {}
        for n in names:
            out["phase"][ph][n] = dict(decision_features([runs[n][k] for k in maps], ph),
                                       dec=sum(decision_counts(runs[n][k], ph)["dec"] for k in maps))
        for k in PHASE_KEYS:
            print(f"{k:20s}" + "".join(fmt(out["phase"][ph][n][k]) for n in names))
    if len(names) >= 2:
        a, b = names[0], names[1]
        print(f"\n-- paired 95% bootstrap CI over maps, {b} minus {a}")
        out["ci"] = {}
        for key in ["won", "dE", "dA", "die_first", "kill_first"]:
            out["ci"][key] = boot_diff(runs[a], runs[b], maps, key)
            print(f"{key:28s} [{out['ci'][key][0]:+.3f}, {out['ci'][key][1]:+.3f}]")
        for num, den, ph in [("mv_away", "mv_vis", None), ("low_mv_away", "low_mv", None), ("focus_all", "focus_steps", None),
                             ("att_weakest", "att", None), ("vis_alone", "vis", None), ("invalid", "dec", None),
                             ("focus_all", "focus_steps", "behind"), ("vis_alone", "vis", "behind"),
                             ("mv_away", "mv_vis", "behind")]:
            ci = boot_ratio(runs[a], runs[b], maps, num, den, ph)
            out["ci"][f"{num}/{den}@{ph}"] = ci
            print(f"{num + '/' + den + '@' + str(ph):28s} [{ci[0]:+.3f}, {ci[1]:+.3f}]")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
