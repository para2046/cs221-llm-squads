"""Robustness checks for action_msg_shift.py on existing logs (no new games).

1. Per-game means (each map weighted equally) vs pooled decisions, for the Table 2 rows, with paired bootstrap
   over maps.
2. Non-exclusive "message mentions an enemy id" rate (first-match categories put "finishing e2" under kill).
3. MAPPO checkpoint evaluations during training: the same 16 dev maps every 2 rounds, T=0.7 (unlike training
   games: random maps, T=1.0). Slope per 10 rounds.
4. Random sample of messages per category for a manual audit of the keyword taxonomy.

usage: python scripts/behavior_robustness.py [--json out.json] [--audit out.txt]
"""
import collections, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import action_msg_shift as A  # noqa: E402
import behavior_shift as BS  # noqa: E402

ROWS = [("enemy_far", "move_toward"), ("enemy_far", "move_away"), ("enemy_far", "invalid"), ("no_enemy", "stop"),
        ("low_hp", "attack_weakest"), ("low_hp", "move_away"), ("in_range_cooling", "move_away")]
MSGS = ["none", "target_named", "kill", "movement", "retreat", "kiting"]
ENEMY = re.compile(r"\be\d+\b", re.I)


def per_game(steps):
    c = A.counts([steps])
    r = {}
    for s, a in ROWS:
        r[f"{s}/{a}"] = c[("sa", s, a)] / c[("sit", s)] if c[("sit", s)] else float("nan")
    for m in MSGS:
        r[f"msg/{m}"] = c[("msg", m)] / c["n"]
    r["msg/retreat_or_kiting"] = r["msg/retreat"] + r["msg/kiting"]
    r["saydo"] = c["saydo"] / c["named"] if c["named"] else float("nan")
    ment = n = 0
    for recs in steps:
        for p, rep, v in recs:
            if A.classify(p, rep, v) is None:
                continue
            n += 1
            ment += bool(ENEMY.search(A.msg_of(rep) or ""))
    r["mentions_enemy"] = ment / n if n else float("nan")
    return r


def main():
    games = {n: BS.load(s) for n, s in A.TEST}
    maps = sorted(set.intersection(*(set(g) for g in games.values())))
    rows = {n: {m: per_game(games[n][m][1]) for m in maps} for n in games}
    names = list(games)
    keys = list(next(iter(rows[names[0]].values())))
    out = {"per_game_means": {}, "change": {}}
    print(f"per-game means over {len(maps)} maps (nan maps skipped)")
    print(f"{'':32s}" + "".join(f"{n:>10s}" for n in names))
    for k in keys:
        vals = [BS.nanmean([rows[n][m][k] for m in maps]) for n in names]
        out["per_game_means"][k] = dict(zip(names, vals))
        print(f"{k:32s}" + "".join(f"{v:10.3f}" for v in vals))
    for a, b in zip(names, names[1:]):
        print(f"-- {a} -> {b} (paired bootstrap over maps, * = excludes 0)")
        out["change"][f"{a}->{b}"] = {}
        for k in keys:
            lo, hi = BS.boot(rows[a], rows[b], maps, k)
            d = BS.nanmean([rows[b][m][k] for m in maps]) - BS.nanmean([rows[a][m][k] for m in maps])
            out["change"][f"{a}->{b}"][k] = dict(diff=d, lo=lo, hi=hi)
            print(f"   {k:30s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")

    by = A.load_mappo("runs/mappo_full/eval_episodes.jsonl")
    its = sorted(by)
    print(f"\n== MAPPO checkpoint evals (16 fixed dev maps, T=0.7), iters {its}")
    out["eval_traj"] = {}
    tabs = {i: A.table(A.counts([g for g, _ in by[i]])) for i in its}
    wins = {i: sum(w for _, w in by[i]) / len(by[i]) for i in its}
    series = [("win", lambda i: wins[i])] + [(f"{s}/{a}", (lambda s, a: lambda i: tabs[i]["act_given_sit"][s][a])(s, a))
                                             for s, a in ROWS] + \
             [(f"msg/{m}", (lambda m: lambda i: tabs[i]["msg"][m])(m)) for m in MSGS]
    for name, f in series:
        ys = [f(i) for i in its]
        b, lo, hi = A.slope(its, ys)
        out["eval_traj"][name] = dict(values=ys, per10=10 * b, lo=10 * lo, hi=10 * hi)
        print(f"  {name:28s} " + " ".join(f"{y:.2f}" for y in ys) +
              f"  slope/10 {10 * b:+.3f} [{10 * lo:+.3f}, {10 * hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")

    if "--audit" in sys.argv:
        rnd = random.Random(1)
        pool = collections.defaultdict(list)
        for n in games:
            for m in maps:
                for recs in games[n][m][1]:
                    for p, rep, v in recs:
                        msg = A.msg_of(rep)
                        if msg and msg.strip():
                            pool[A.msg_cat(msg)].append((n, msg.strip()))
        with open(sys.argv[sys.argv.index("--audit") + 1], "w") as fo:
            for cat in A.MSG_CATS:
                if cat == "none" or not pool[cat]:
                    continue
                fo.write(f"== {cat} ({len(pool[cat])} messages)\n")
                for n, msg in rnd.sample(pool[cat], min(25, len(pool[cat]))):
                    fo.write(f"  [{n}] {msg}\n")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
