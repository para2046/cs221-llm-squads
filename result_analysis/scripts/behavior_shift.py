"""How does behaviour change from untrained E4B -> SFT on 31B games -> MAPPO, and which changes go with winning?

1. Per-game behaviour features (scripts/win_loss_analysis.features + a few more) for every model on the same maps.
2. Paired change between consecutive models (bootstrap 95% CI over maps).
3. Which features predict winning: logistic regression on all logged games vs the built-in AI, one intercept per
   player (so it compares won and lost games of the same player), standardised features.
4. Decomposition: predicted change in log-odds of winning = sum_k beta_k * (mean change in feature k). The part
   that the features do not explain is reported as a remainder. This is correlational, not causal.

usage: python scripts/behavior_shift.py [--maps dev|new] [--phase all|opening|after] [--json out.json]
"opening" = steps before either side loses a unit, so the features are less driven by already winning or losing.
"""
import collections, glob, json, math, random, re, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import win_loss_analysis as W  # noqa: E402

MATE = re.compile(r"^\s+a(\d) (\w+) at \(([-\d.]+), ([-\d.]+)\), dist ([\d.]+)", re.M)
DIRS = {"move_north": (0, 1), "move_south": (0, -1), "move_east": (1, 0), "move_west": (-1, 0)}
MODELS = {
    "dev": [("untrained", "runs/e2u_4b_struct/episodes.jsonl"), ("SFT", "runs/e4_awr_c2/episodes.jsonl")],
    "new": [("untrained", "runs/f64_untrained/episodes.jsonl"), ("SFT", "runs/f64_sft/eval_episodes.jsonl"),
            ("MAPPO", "runs/f64_mappo/eval_episodes.jsonl"),
            ("31B", "runs/evalonly_31b_fixed40/w*.jsonl|runs/evalonly_31b_fixed64rest/w*.jsonl")],
}


def load(spec):
    """{scenario: [steps]} where steps = list (per t) of [(prompt, reply, valid)]."""
    out = {}
    for pat in spec.split("|"):
        for f in sorted(glob.glob(pat)):
            if "/w" in f.split("runs/")[1] and f.endswith(".jsonl") and "episodes" not in f:   # play.py logs
                eps, scen, bad = collections.defaultdict(list), {}, collections.Counter()
                for line in open(f):
                    x = json.loads(line)
                    if x.get("summary"):
                        scen[x["episode"]] = (x.get("scenario"), bool(x["won"]))
                        continue
                    bad[x["episode"]] += sum((r.get("reply") or "").startswith("ERROR") for r in x["team_a"])
                    eps[x["episode"]].append([(r["prompt"], r.get("reply") or "", r.get("valid", True)) for r in x["team_a"]])
                for e, steps in eps.items():
                    if e in scen and scen[e][0] is not None and not bad[e]:
                        out[scen[e][0]] = (scen[e][1], steps)
            else:
                for line in open(f):
                    x = json.loads(line)
                    if x.get("team", "A") != "A" or x.get("scenario") is None or x["scenario"] in out:
                        continue
                    by_t = collections.defaultdict(list)
                    for r in x["decisions"]:
                        by_t[int(r["t"])].append((r["prompt"], r.get("reply") or "", r.get("valid", True)))
                    out[x["scenario"]] = (bool(x["info"].get("battle_won")), [by_t[t] for t in sorted(by_t)])
    return out


def extra(steps):
    """Features not in win_loss_analysis: disengaging, isolation, invalid orders, messages, who loses a unit first."""
    c = collections.Counter()
    for recs in steps:
        for prompt, reply, valid in recs:
            m = W.SELF.search(prompt)
            if not m:
                continue
            x, y = float(m.group(3)), float(m.group(4))
            hp, hpm, sh, shm = int(m.group(5)), int(m.group(6)), int(m.group(7) or 0), int(m.group(8) or 0)
            body = prompt.split("Units alive")[0]
            en = [(float(e.group(3)), float(e.group(4)), float(e.group(5))) for e in W.ENEMY.finditer(body)]
            tm = prompt.split("Visible teammates:")[1].split("Visible enemies")[0] if "Visible teammates:" in prompt else ""
            a = W.action_of(reply)
            c["n"] += 1
            c["invalid"] += not valid
            try:
                msg = (json.loads(reply).get("msg") or "").strip()
            except Exception:
                mm = re.search(r'"msg"\s*:\s*"([^"]*)"', reply or "")
                msg = mm.group(1).strip() if mm else ""
            c["msg"] += bool(msg)
            c["msg_named"] += bool(re.search(r"\be\d\b", msg.lower()))
            if en:
                c["vis"] += 1
                c["alone"] += not MATE.search(tm)
                if a in DIRS:
                    ex, ey, _ = min(en, key=lambda e: e[2])
                    dx, dy = DIRS[a]
                    c["mv"] += 1
                    c["away"] += (ex - x) * dx + (ey - y) * dy <= 0
                    if (hp + sh) < 0.35 * (hpm + shm):
                        c["low_mv"] += 1
                        c["low_away"] += (ex - x) * dx + (ey - y) * dy <= 0
                c["stop"] += a == "stop"
    r = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    return {"invalid": r("invalid", "n"), "msg_rate": r("msg", "n"), "msg_named": r("msg_named", "n"),
            "alone_with_enemy": r("alone", "vis"), "move_away": r("away", "mv"), "low_hp_away": r("low_away", "low_mv"),
            "stop_with_enemy": r("stop", "vis")}


def opening(steps):
    """Steps before either side loses a unit (behaviour not yet shaped by being ahead or behind)."""
    out = []
    for recs in steps:
        m = next((W.ALIVE.search(p) for p, _, _ in recs if W.ALIVE.search(p)), None)
        if m and (int(m.group(1)) < 5 or int(m.group(2)) < 5):
            break
        out.append(recs)
    return out


def game_row(won, steps, phase="all"):
    if phase == "opening":
        steps = opening(steps)
    elif phase == "after":
        steps = steps[len(opening(steps)):]
    f = W.features([[(p, rep) for p, rep, _ in recs] for recs in steps])
    f.update(extra(steps))
    f["won"] = float(won)
    return f


SHOW = ["won", "first_blood", "attack", "focus", "targets_per_step", "weakest", "target_switch", "ready_noshoot",
        "cd_ir_move", "low_hp_move", "low_hp_away", "move_away", "stop_with_enemy", "stalker_kite", "zealot_attack",
        "sync_at_contact", "cohesion_pre", "cohesion_fight", "regroup", "alone_with_enemy", "invalid", "msg_rate",
        "msg_named"]
REG = ["focus", "weakest", "target_switch", "cd_ir_move", "low_hp_move", "move_away", "sync_at_contact",
       "cohesion_fight", "alone_with_enemy", "invalid", "spawn_spread", "colossi", "zealots"]


def nanmean(xs):
    xs = [v for v in xs if not math.isnan(v)]
    return sum(xs) / len(xs) if xs else float("nan")


def boot(a, b, maps, k, B=2000):
    rnd = random.Random(0)
    d = [b[m][k] - a[m][k] for m in maps if not (math.isnan(a[m][k]) or math.isnan(b[m][k]))]
    if not d:
        return float("nan"), float("nan")
    s = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(B))
    return s[int(.025 * B)], s[int(.975 * B)]


PHASE = ["all"]


def fit_win_model():
    """Logit with player fixed effects on all logged games vs the built-in AI (win_loss_analysis sources) plus ours."""
    rows = []
    for src, d, won, steps in W.load_games():
        if steps:
            rows.append((src, game_row(won, [[(p, r, True) for p, r in recs] for recs in steps], phase=PHASE[0])))
    return rows


def main():
    which = sys.argv[sys.argv.index("--maps") + 1] if "--maps" in sys.argv else "new"
    models = [(n, load(s)) for n, s in MODELS[which] if glob.glob(s.split("|")[0])]
    maps = sorted(set.intersection(*(set(g) for _, g in models)))
    PH = sys.argv[sys.argv.index("--phase") + 1] if "--phase" in sys.argv else "all"
    PHASE[0] = PH
    feats = {n: {m: game_row(*g[m], phase=PH) for m in maps} for n, g in models}
    for n, g in models:   # outcome-level columns always from the whole game
        for m in maps:
            whole = game_row(*g[m]) if PH != "all" else feats[n][m]
            for k in ("won", "first_blood", "spawn_spread", "colossi", "zealots"):
                feats[n][m][k] = whole[k]
    names = [n for n, _ in models]
    print(f"maps: {which}, {len(maps)} shared; models: {names}; behaviour phase: {PH}")
    print(f"{'feature':20s}" + "".join(f"{n:>11s}" for n in names))
    for k in SHOW:
        print(f"{k:20s}" + "".join(f"{nanmean([feats[n][m][k] for m in maps]):11.3f}" for n in names))
    out = {"maps": which, "n_maps": len(maps), "means": {n: {k: nanmean([feats[n][m][k] for m in maps]) for k in SHOW}
                                                          for n in names}, "change": {}, "decomp": {}}
    pairs = [(a, b) for a, b in [("untrained", "SFT"), ("SFT", "MAPPO"), ("MAPPO", "31B")] if a in names and b in names]
    for a, b in pairs:
        print(f"\n-- {a} -> {b}: paired change, 95% bootstrap CI over maps (* = CI excludes 0)")
        out["change"][f"{a}->{b}"] = {}
        for k in SHOW:
            lo, hi = boot(feats[a], feats[b], maps, k)
            dm = nanmean([feats[b][m][k] for m in maps]) - nanmean([feats[a][m][k] for m in maps])
            star = "*" if lo > 0 or hi < 0 else " "
            out["change"][f"{a}->{b}"][k] = dict(diff=dm, lo=lo, hi=hi)
            print(f"  {k:20s} {dm:+.3f}  [{lo:+.3f}, {hi:+.3f}] {star}")

    # win model: pooled other players + our models' games (one fixed effect per player)
    pool = fit_win_model()
    for n in names:
        pool += [(f"ours_{n}", feats[n][m]) for m in maps]
    sel = [(s, f) for s, f in pool if not any(math.isnan(f[k]) for k in REG)]
    X = np.array([[f[k] for k in REG] for _, f in sel])
    mu, sd = X.mean(0), np.where(X.std(0) > 0, X.std(0), 1)
    y = np.array([f["won"] for _, f in sel])
    beta, se = W.logit_fe((X - mu) / sd, y, [s for s, _ in sel])
    print(f"\n-- win model: {len(sel)} games, {len(set(s for s, _ in sel))} players (fixed effects); OR per +1 SD")
    out["win_model"] = {}
    for k, b, s in zip(REG, beta, se):
        out["win_model"][k] = dict(OR=float(np.exp(b)), z=float(b / s), sd=float(sd[REG.index(k)]))
        print(f"  {k:20s} OR {np.exp(b):5.2f}  z {b / s:6.2f}")
    for a, b in pairs:
        ma = {k: nanmean([feats[a][m][k] for m in maps]) for k in REG}
        mb = {k: nanmean([feats[b][m][k] for m in maps]) for k in REG}
        contrib = {k: float(beta[i] * (mb[k] - ma[k]) / sd[i]) for i, k in enumerate(REG)}
        wa, wb = out["means"][a]["won"], out["means"][b]["won"]
        lo = lambda p: math.log(max(p, 1e-3) / max(1 - p, 1e-3))
        total = lo(wb) - lo(wa)
        out["decomp"][f"{a}->{b}"] = dict(actual_logodds=total, explained=sum(contrib.values()), contrib=contrib)
        print(f"\n-- {a} -> {b}: win {wa:.3f} -> {wb:.3f}, log-odds change {total:+.2f}; "
              f"explained by behaviour features {sum(contrib.values()):+.2f}")
        for k, v in sorted(contrib.items(), key=lambda kv: -abs(kv[1])):
            print(f"  {k:20s} {v:+.2f}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
