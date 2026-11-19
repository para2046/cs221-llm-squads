"""Per-game battle form: how each fight unfolded, from the per-unit prompts in an episode log.

usage: python scripts/battle_forms.py NAME=FILE [NAME=FILE ...] [--csv out.csv]
Works with train.py (episodes.jsonl) and mappo.py (eval_episodes.jsonl) logs; team A only.
"""
import json, re, sys, math, itertools
from collections import Counter, defaultdict

sys.path.insert(0, "scripts")
from behavior_stats import parse, action_of, DIRS  # noqa: E402

ALIVE = re.compile(r"Units alive: your team (\d+), enemy (\d+)")
ALLY = re.compile(r"^\s+a(\d) (\w+) at \(([-\d.]+), ([-\d.]+)\)", re.M)


def game(ep):
    by_t = defaultdict(list)
    for d in ep["decisions"]:
        by_t[int(d["t"])].append(d)
    T = sorted(by_t)
    comp, etypes = Counter(), {}
    alive_tl, pos_tl = {}, {}
    contact = None
    att = foc_n = foc = away = mv_vis = 0
    dec = 0
    zealot_wait = zealot_n = 0
    for t in T:
        ds = by_t[t]
        alive = None
        pos = {}
        targets = []
        for d in ds:
            p = d["prompt"]
            me, en = parse(p)
            m = ALIVE.search(p)
            if m:
                alive = (int(m[1]), int(m[2]))
            pos[d["agent"]] = (me["x"], me["y"])
            if t == T[0]:
                comp[me["typ"]] += 1
            for e in en:
                etypes[e["id"]] = e["typ"]
            if en and contact is None:
                contact = t
            act, _ = action_of(d["reply"])
            dec += 1
            inr = [e for e in en if e["inr"]]
            if act.startswith("attack"):
                att += 1
                if inr:
                    targets.append(act.split("_", 1)[1])
            if en and act in DIRS:
                near = min(en, key=lambda e: e["d"])
                dx, dy = DIRS[act]
                mv_vis += 1
                away += (near["x"] - me["x"]) * dx + (near["y"] - me["y"]) * dy <= 0
            if me["typ"] == "zealot" and en:
                zealot_n += 1
                zealot_wait += not act.startswith("attack") and not inr and act in DIRS and \
                    (min(en, key=lambda e: e["d"])["x"] - me["x"]) * DIRS[act][0] + \
                    (min(en, key=lambda e: e["d"])["y"] - me["y"]) * DIRS[act][1] <= 0
        if len(targets) >= 2:
            foc_n += 1
            foc += Counter(targets).most_common(1)[0][1] / len(targets)
        alive_tl[t] = alive
        pos_tl[t] = pos
    info = ep["info"]
    won = bool(info.get("battle_won"))
    dA, dE = info.get("dead_allies", 0), info.get("dead_enemies", 0)
    steps = len(ep["rewards"])
    # spread of our squad at first contact (max pairwise distance)
    c = contact if contact is not None else T[-1]
    pts = list(pos_tl.get(c, {}).values())
    spread = max((math.dist(a, b) for a, b in itertools.combinations(pts, 2)), default=0.0)
    # who lost a unit first, and the trade 10 steps after contact
    first_blood, trade10 = None, None
    prev = (5, 5)
    for t in T:
        a = alive_tl[t]
        if a is None:
            continue
        if first_blood is None and (a[0] < prev[0] or a[1] < prev[1]):
            first_blood = "us" if a[0] < prev[0] and a[1] == prev[1] else ("them" if a[1] < prev[1] and a[0] == prev[0] else "both")
        if trade10 is None and contact is not None and t >= contact + 10:
            trade10 = (5 - a[0], 5 - a[1])
        prev = a
    if steps >= 200:
        form = "timeout"
    elif won:
        form = "stomp win" if dA <= 2 else "close win"
    else:
        form = "close loss" if dE >= 4 else ("collapse" if dE <= 2 else "loss")
    return dict(scenario=ep["scenario"], won=int(won), form=form, steps=steps, dA=dA, dE=dE,
                comp="".join(f"{v}{k[0].upper()}" for k, v in sorted(comp.items())),
                enemy="".join(f"{v}{k[0].upper()}" for k, v in sorted(Counter(etypes.values()).items())),
                contact=contact, spread=round(spread, 1), first_blood=first_blood,
                trade10=trade10, focus=round(foc / max(foc_n, 1), 2), attack=round(att / max(dec, 1), 2),
                away=round(away / max(mv_vis, 1), 2), zealot_backoff=round(zealot_wait / max(zealot_n, 1), 3))


def load(path):
    out = {}
    for l in open(path):
        ep = json.loads(l)
        if ep.get("team", "A") in ("A", "1", 1) or "team" not in ep:
            out[ep["scenario"]] = game(ep)
    return out


def main():
    args = [a for a in sys.argv[1:] if "=" in a]
    runs = {k: load(v) for k, v in (a.split("=", 1) for a in args)}
    names = list(runs)
    for n in names:
        g = runs[n].values()
        forms = Counter(x["form"] for x in g)
        fb = Counter(x["first_blood"] for x in g)
        mean = lambda k: round(sum(x[k] for x in g if x[k] is not None) / len(g), 2)
        print(f"== {n}: {len(g)} games, win {sum(x['won'] for x in g)}  forms {dict(forms)}  first blood {dict(fb)}")
        print(f"   contact {mean('contact')}  spread@contact {mean('spread')}  focus {mean('focus')}  "
              f"away {mean('away')}  zealot backoff {mean('zealot_backoff')}  steps {mean('steps')}")
        for fb_side in ("us", "them", "both"):
            sub = [x for x in g if x["first_blood"] == fb_side]
            if sub:
                print(f"   first blood {fb_side:4s}: {len(sub):2d} games, win {sum(x['won'] for x in sub)/len(sub):.2f}")
    if len(names) == 2:
        a, b = names
        print(f"\n== per scenario where outcomes differ ({a} vs {b})")
        keys = ["form", "comp", "enemy", "contact", "spread", "first_blood", "trade10", "focus", "away", "zealot_backoff", "dA", "dE"]
        for s in sorted(set(runs[a]) & set(runs[b])):
            x, y = runs[a][s], runs[b][s]
            if x["won"] != y["won"]:
                print(s, f"{a}:", {k: x[k] for k in keys})
                print(" " * len(str(s)), f"{b}:", {k: y[k] for k in keys})


if __name__ == "__main__":
    main()
