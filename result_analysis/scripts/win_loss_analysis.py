"""What separates won from lost games? Per-game behaviour features (team A, vs the built-in AI) computed from the
prompts/replies of every logged game, compared within each player (source) and pooled with source fixed effects.

usage: python scripts/win_loss_analysis.py [--json out.json]
"""
import collections
import glob
import json
import math
import re
import sys

import numpy as np

SOURCES = {  # name: glob of run dirs (vs built-in AI only)
    "rule_focus_local": ["runs/rule_focus_local"], "rule_kite_local": ["runs/rule_kite_local"],
    "rule_kite_global": ["runs/rule_kite", "runs/rule_scripted"],
    "gemma31b": ["runs/teacher_gemma31b", "runs/teacher31b_strategy"],
    "luna": ["runs/luna56_strategy"],
    "gptoss": ["runs/d12_gptoss120b_builtin_none", "runs/d12_gptoss120b_builtin_struct",
               "runs/d12_gptoss120b_rules_none", "runs/d12_gptoss120b_rules_struct2"],
    "inkling": ["runs/d12_inkling2small_builtin_none", "runs/d12_inkling2small_builtin_struct",
                "runs/d12_inkling2small_rules_none", "runs/d12_inkling2small_rules_struct2", "runs/ink40_*"],
    "gemma4b": ["runs/e2_4b_*", "runs/sp_eval_base"],
}
RANGE = {"zealot": 2, "stalker": 6, "colossus": 7}
SELF = re.compile(r"You are agent a(\d+), a (\w+) at \(([\d.-]+), ([\d.-]+)\), hp (\d+)/(\d+)(?:, shield (\d+)/(\d+))?, "
                  r"weapon cooldown (\d+)")
ENEMY = re.compile(r"^\s+e(\d+) (\w+) at \(([\d.-]+), ([\d.-]+)\), dist ([\d.]+), hp (\d+)/(\d+)(?:, shield (\d+)/(\d+))?"
                   r"( \[in range\])?", re.M)
ALIVE = re.compile(r"Units alive: your team (\d+), enemy (\d+)")
ACT = re.compile(r'"action"\s*:\s*"([a-z_0-9]+)"')


def load_games():
    """Yield (source, run_dir, won, steps) where steps = list of lists of (prompt, action) for team A."""
    for src, pats in SOURCES.items():
        for pat in pats:
            for d in sorted(glob.glob(pat)):
                for f in sorted(glob.glob(d + "/episodes.jsonl")):          # smac_llm.train format
                    for line in open(f):
                        x = json.loads(line)
                        if x.get("team") == "B":
                            continue
                        by_t = collections.defaultdict(list)
                        for r in x["decisions"]:
                            by_t[r["t"]].append((r["prompt"], r.get("reply") or ""))
                        yield src, d, bool(x["info"].get("battle_won")), [by_t[t] for t in sorted(by_t)]
                for f in sorted(glob.glob(d + "/w*.jsonl")):                # smac_llm.run / smac_llm.play format
                    eps = collections.defaultdict(list)
                    won = {}
                    for line in open(f):
                        x = json.loads(line)
                        if x.get("summary"):
                            won[x["episode"]] = bool(x["won"])
                            continue
                        recs = x.get("team_a") or x.get("agents") or []
                        eps[x["episode"]].append([(r.get("prompt") or "", r.get("reply") or "") for r in recs])
                    for e, steps in eps.items():
                        if e in won:
                            yield src, d, won[e], steps


def action_of(reply):
    m = ACT.search(reply or "")
    return m.group(1) if m else ""


def features(steps):
    c = collections.Counter()
    first_contact = first_kill = first_death = None
    prev_alive = None
    cohesion_pre, cohesion_fight = [], []
    prev_target = {}
    for t, recs in enumerate(steps):
        pos, tgts, contact_units = [], [], 0
        alive_line = None
        for prompt, reply in recs:
            m = SELF.search(prompt)
            if not m:
                continue
            aid, typ = int(m.group(1)), m.group(2)
            x, y = float(m.group(3)), float(m.group(4))
            hp, hpm = int(m.group(5)), int(m.group(6))
            sh, shm = int(m.group(7) or 0), int(m.group(8) or 0)
            cd = int(m.group(9))
            frac = (hp + sh) / max(hpm + shm, 1)
            pos.append((x, y))
            body = prompt.split("Units alive")[0]
            en = [(int(e.group(1)), e.group(2), float(e.group(5)), int(e.group(6)) + int(e.group(8) or 0),
                   bool(e.group(10))) for e in ENEMY.finditer(body)]
            in_range = [e for e in en if e[4]]
            near = min((e[2] for e in en), default=99)
            if near <= RANGE.get(typ, 6) + 3:
                contact_units += 1
            a = action_of(reply)
            c["n"] += 1
            if in_range:
                c["n_ir"] += 1
            if a.startswith("attack_e"):
                k = int(a[8:])
                c["attack"] += 1
                tgts.append(k)
                if in_range and k in [e[0] for e in in_range]:
                    c["att_ir"] += 1
                    c["weakest"] += [e for e in in_range if e[0] == k][0][3] == min(e[3] for e in in_range)
                if aid in prev_target and prev_target[aid] != k and prev_target[aid] in [e[0] for e in in_range]:
                    c["switch"] += 1
                if aid in prev_target and prev_target[aid] in [e[0] for e in in_range]:
                    c["switch_n"] += 1
                prev_target[aid] = k
            if cd == 0 and in_range:
                c["ready_ir"] += 1
                c["ready_noshoot"] += not a.startswith("attack_e")
            if cd > 0 and in_range:
                c["cd_ir"] += 1
                c["cd_ir_move"] += a.startswith("move")
            if frac < 0.35 and in_range:
                c["low_ir"] += 1
                c["low_ir_move"] += a.startswith("move")
            if typ == "stalker" and cd > 0 and any(e[1] == "zealot" and e[2] < 4 for e in en):
                c["kite_n"] += 1
                c["kite_move"] += a.startswith("move")
            if typ == "zealot" and in_range:
                c["zeal_ir"] += 1
                c["zeal_ir_att"] += a.startswith("attack_e")
            m2 = ALIVE.search(prompt)
            if m2:
                alive_line = (int(m2.group(1)), int(m2.group(2)))
        if len(tgts) >= 2:
            c["att_steps"] += 1
            c["focus"] += len(set(tgts)) == 1
            c["targets_sum"] += len(set(tgts))
        if len(pos) >= 2:
            cx, cy = sum(p[0] for p in pos) / len(pos), sum(p[1] for p in pos) / len(pos)
            coh = sum(math.dist(p, (cx, cy)) for p in pos) / len(pos)
            (cohesion_pre if first_contact is None else cohesion_fight).append(coh)
        if first_contact is None and contact_units:
            first_contact = t
            c["sync"] = contact_units / max(len(pos), 1)   # share of units near the enemy at first contact
        if alive_line and prev_alive:
            if alive_line[1] < prev_alive[1] and first_kill is None:
                first_kill = t
            if alive_line[0] < prev_alive[0] and first_death is None:
                first_death = t
        if alive_line:
            prev_alive = alive_line
    r = lambda a, b: c[a] / c[b] if c[b] else float("nan")
    return {
        "attack": r("attack", "n"),
        "focus": r("focus", "att_steps"),
        "targets_per_step": r("targets_sum", "att_steps"),
        "weakest": r("weakest", "att_ir"),
        "ready_noshoot": r("ready_noshoot", "ready_ir"),      # weapon ready, enemy in range, but did not attack
        "cd_ir_move": r("cd_ir_move", "cd_ir"),              # weapon cooling, enemy in range: moved (step back)
        "low_hp_move": r("low_ir_move", "low_ir"),           # below 35% hp+shield with enemy in range: moved
        "stalker_kite": r("kite_move", "kite_n"),            # stalker cooling with a zealot within 4: moved
        "zealot_attack": r("zeal_ir_att", "zeal_ir"),
        "target_switch": r("switch", "switch_n"),            # left a still-in-range target for another
        "sync_at_contact": c["sync"] if first_contact is not None else float("nan"),
        "cohesion_pre": float(np.mean(cohesion_pre)) if cohesion_pre else float("nan"),
        "cohesion_fight": float(np.mean(cohesion_fight)) if cohesion_fight else float("nan"),
        "spawn_spread": cohesion_pre[0] if cohesion_pre else float("nan"),
        "regroup": (cohesion_pre[0] - cohesion_pre[-1]) if len(cohesion_pre) >= 2 else 0.0,
        "contact_step": float(first_contact) if first_contact is not None else float("nan"),
        "first_blood": (1.0 if (first_kill is not None and (first_death is None or first_kill < first_death))
                        else 0.0 if first_death is not None or first_kill is not None else float("nan")),
        "colossi": float(sum(1 for p, _ in steps[0] if " a colossus at" in p)) if steps else float("nan"),
        "zealots": float(sum(1 for p, _ in steps[0] if " a zealot at" in p)) if steps else float("nan"),
    }


def logit_fe(X, y, groups, l2=1e-3, iters=50):
    """Logistic regression with one intercept per group (source fixed effects); Newton steps."""
    G = sorted(set(groups))
    D = np.array([[g == h for h in G] for g in groups], float)
    Z = np.hstack([D, X])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        H = Z.T @ (Z * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, Z.T @ (p - y) + l2 * w)
    p = 1 / (1 + np.exp(-Z @ w))
    H = Z.T @ (Z * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return w[len(G):], se[len(G):]


def main():
    rows = [(src, d, won, features(steps)) for src, d, won, steps in load_games() if steps]
    keys = list(rows[0][3].keys())
    print(f"{len(rows)} games")
    print("\n== per source: wins / games, and feature mean in WON vs LOST games ==")
    by = collections.defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    show = ["attack", "focus", "weakest", "ready_noshoot", "cd_ir_move", "low_hp_move", "stalker_kite",
            "target_switch", "sync_at_contact", "cohesion_pre", "first_blood"]
    print(f"{'source':18s} {'W/N':>7s} " + " ".join(f"{k[:11]:>13s}" for k in show))
    for src, rs in by.items():
        w = [r for r in rs if r[2]]
        l = [r for r in rs if not r[2]]
        cells = []
        for k in show:
            mw = np.nanmean([r[3][k] for r in w]) if w else float("nan")
            ml = np.nanmean([r[3][k] for r in l]) if l else float("nan")
            cells.append(f"{mw:5.2f}/{ml:5.2f}  ")
        print(f"{src:18s} {len(w):3d}/{len(rs):3d} " + " ".join(f"{c:>13s}" for c in cells))
    print("\n== pooled, source fixed effects, one feature at a time (standardised; odds ratio per +1 SD) ==")
    out = {}
    for k in keys:
        sel = [r for r in rows if not math.isnan(r[3][k])]
        x = np.array([r[3][k] for r in sel])
        if len(sel) < 30 or x.std() == 0:
            continue
        xs = ((x - x.mean()) / x.std())[:, None]
        y = np.array([r[2] for r in sel], float)
        b, se = logit_fe(xs, y, [r[0] for r in sel])
        z = b[0] / se[0]
        out[k] = {"or_per_sd": round(float(np.exp(b[0])), 2), "z": round(float(z), 2), "n": len(sel)}
        print(f"  {k:18s} OR {np.exp(b[0]):5.2f}  z {z:6.2f}  n {len(sel)}")
    print("\n== pooled multivariate (behaviour only, plus composition), source fixed effects ==")
    mk = ["focus", "weakest", "cd_ir_move", "low_hp_move", "target_switch", "sync_at_contact",
          "spawn_spread", "regroup", "colossi", "zealots"]
    sel = [r for r in rows if not any(math.isnan(r[3][k]) for k in mk)]
    X = np.array([[r[3][k] for k in mk] for r in sel])
    X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    y = np.array([r[2] for r in sel], float)
    b, se = logit_fe(X, y, [r[0] for r in sel])
    for k, bb, ss in zip(mk, b, se):
        print(f"  {k:18s} OR {np.exp(bb):5.2f}  z {bb / ss:6.2f}")
    print(f"  (n = {len(sel)} games)")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
