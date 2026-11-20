"""Shift in actions and in message types: untrained E4B -> SFT on 31B games -> MAPPO -> Gemma 4 31B, on the same
64 test maps (200000-200063), plus the trajectory over MAPPO training rounds.

Actions. Every decision is put in a SITUATION (what the unit sees) and an ACTION class:
  situations: no_enemy | enemy_far (visible, none in range) | in_range_ready (cooldown 0) | in_range_cooling |
              low_hp (below 35% hp+shield, enemy visible; overrides the others)
  actions:    attack_weakest | attack_other | move_toward (nearest enemy) | move_away | move (no enemy visible) |
              stop | invalid
  The situation mix differs between models (a better team spends more time in range), so we report P(action |
  situation) and compare models within each situation.
Messages. One category per message, first match wins (scripts/msg_taxonomy.py CATS), plus "none".
  say-do: the unit names enemy e_k in its message and attacks e_k in the same reply.
Training. MAPPO dumps every training game (16 per round, temperature 1.0, random maps) and an evaluation every
  2 rounds (16 fixed dev maps, temperature 0.7). Round r's games come from the policy after r-1 actor updates
  (round 0 trains only the critic); evaluation "round 0" is the SFT policy.

usage: python scripts/action_msg_shift.py [--json out.json] [--fig-dir DIR]
"""
import collections, glob, json, math, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import behavior_shift as BS  # noqa: E402
import win_loss_analysis as W  # noqa: E402

CATS = [  # same order / regexes as scripts/msg_taxonomy.py
    ("backup", r"\b(help|need|assist|support|backup|cover me|join(ing)? (fight|team|you)|coming to)\b"),
    ("regroup", r"\b(regroup|group(ing)? up|gather|wait|together|hold)\b"),
    ("retreat", r"\b(retreat\w*|fall(ing)? back|back(ing)? off|low (hp|health|shield)|recover\w*|dying)\b"),
    ("kiting", r"\bkit(e|ing)\b"),
    ("kill", r"\b(finish\w*|kill(ed)?|down|dead|last hit)\b"),
    ("target_named", r"\be\d\b"),
    ("target_type", r"\b(focus\w*|target\w*|attack\w*|burn\w*|fire)\b"),
    ("movement", r"\b(mov\w*|head\w*|go\w*|advanc\w*|approach\w*|search\w*|scout\w*|north|south|east|west|closing)\b"),
]
MSG_CATS = ["none"] + [c for c, _ in CATS] + ["other"]
SITS = ["no_enemy", "enemy_far", "in_range_ready", "in_range_cooling", "low_hp"]
ACTS = ["attack_weakest", "attack_other", "move_toward", "move_away", "move", "stop", "invalid"]
DIRS = BS.DIRS
TEST = [("untrained", "runs/f64_untrained/episodes.jsonl"), ("SFT", "runs/f64_sft/eval_episodes.jsonl"),
        ("MAPPO", "runs/f64_mappo/eval_episodes.jsonl"),
        ("31B", "runs/evalonly_31b_fixed40/w*.jsonl|runs/evalonly_31b_fixed64rest/w*.jsonl")]


def msg_of(reply):
    try:
        return (json.loads(reply).get("msg") or "").strip()
    except Exception:
        m = re.search(r'"msg"\s*:\s*"([^"]*)"', reply or "")
        return m.group(1).strip() if m else ""


def msg_cat(m):
    if not m:
        return "none"
    t = m.lower()
    for n, r in CATS:
        if re.search(r, t):
            return n
    return "other"


def classify(prompt, reply, valid):
    m = W.SELF.search(prompt)
    if not m:
        return None
    x, y = float(m.group(3)), float(m.group(4))
    hp, hpm, sh, shm, cd = int(m.group(5)), int(m.group(6)), int(m.group(7) or 0), int(m.group(8) or 0), int(m.group(9))
    body = prompt.split("Units alive")[0]
    en = [dict(id=f"e{e.group(1)}", x=float(e.group(3)), y=float(e.group(4)), d=float(e.group(5)),
               hs=int(e.group(6)) + int(e.group(8) or 0), inr=bool(e.group(10))) for e in W.ENEMY.finditer(body)]
    inr = [e for e in en if e["inr"]]
    if en and (hp + sh) < 0.35 * (hpm + shm):
        sit = "low_hp"
    elif not en:
        sit = "no_enemy"
    elif not inr:
        sit = "enemy_far"
    else:
        sit = "in_range_ready" if cd == 0 else "in_range_cooling"
    a = W.action_of(reply)
    if not valid or not a:
        act = "invalid"
    elif a.startswith("attack_e"):
        tgt = a[7:]
        act = "attack_weakest" if inr and tgt == min(inr, key=lambda e: e["hs"])["id"] else "attack_other"
    elif a in DIRS:
        if en:
            n = min(en, key=lambda e: e["d"])
            dx, dy = DIRS[a]
            act = "move_toward" if (n["x"] - x) * dx + (n["y"] - y) * dy > 0 else "move_away"
        else:
            act = "move"
    elif a == "stop":
        act = "stop"
    else:
        act = "invalid"
    msg = msg_of(reply)
    named = set(re.findall(r"\be\d\b", msg.lower()))
    sd = None
    if named:
        sd = a.startswith("attack_e") and a[7:] in named
    return sit, act, msg_cat(msg), sd


def counts(games):
    """games: iterable of steps (list per t of [(prompt, reply, valid)]). Returns Counters."""
    c = collections.Counter()
    for steps in games:
        for recs in steps:
            for p, r, v in recs:
                k = classify(p, r, v)
                if k is None:
                    continue
                sit, act, mc, sd = k
                c["n"] += 1
                c[("sit", sit)] += 1
                c[("act", act)] += 1
                c[("sa", sit, act)] += 1
                c[("msg", mc)] += 1
                if sd is not None:
                    c["named"] += 1
                    c["saydo"] += sd
    return c


def table(c):
    n = c["n"]
    out = {"n": n, "sit": {s: c[("sit", s)] / n for s in SITS}, "act": {a: c[("act", a)] / n for a in ACTS},
           "act_given_sit": {s: {a: c[("sa", s, a)] / max(c[("sit", s)], 1) for a in ACTS} for s in SITS},
           "msg": {m: c[("msg", m)] / n for m in MSG_CATS}}
    nm = n - c[("msg", "none")]
    out["msg_among_msgs"] = {m: c[("msg", m)] / max(nm, 1) for m in MSG_CATS if m != "none"}
    out["saydo"] = c["saydo"] / max(c["named"], 1)
    return out


def jsd(p, q):
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}
    kl = lambda a: sum(a.get(k, 0) * math.log2(a.get(k, 0) / m[k]) for k in keys if a.get(k, 0) > 0)
    return (kl(p) + kl(q)) / 2


def boot_cell(ca, cb, maps, key_num, key_den, B=1000):
    """paired bootstrap over maps of the change (b - a) in c[key_num]/c[key_den]."""
    rnd = random.Random(0)
    d = []
    for _ in range(B):
        s = [rnd.choice(maps) for _ in maps]
        A, Bc = collections.Counter(), collections.Counter()
        for k in s:
            A.update(ca[k]); Bc.update(cb[k])
        d.append(Bc[key_num] / max(Bc[key_den], 1) - A[key_num] / max(A[key_den], 1))
    d.sort()
    return d[int(.025 * B)], d[int(.975 * B)]


def load_mappo(path):
    """{iter: [steps per game]} from a mappo.py episode dump (team A)."""
    by = collections.defaultdict(list)
    for l in open(path):
        e = json.loads(l)
        if e.get("team", "A") != "A":
            continue
        T = collections.defaultdict(list)
        for d in e["decisions"]:
            T[int(d["t"])].append((d["prompt"], d.get("reply") or "", d.get("valid", True)))
        by[e["iter"]].append(([T[t] for t in sorted(T)], bool(e["info"].get("battle_won"))))
    return by


def slope(xs, ys):
    """OLS slope and a 95% bootstrap interval over points."""
    def ols(px, py):
        mx, my = sum(px) / len(px), sum(py) / len(py)
        den = sum((a - mx) ** 2 for a in px)
        return sum((a - mx) * (b - my) for a, b in zip(px, py)) / den if den else 0.0
    rnd = random.Random(0)
    pts = list(zip(xs, ys))
    bs = sorted(ols(*zip(*[rnd.choice(pts) for _ in pts])) for _ in range(1000))
    return ols(xs, ys), bs[25], bs[975]


TRACK = [("act_given_sit", "low_hp", "move_away"), ("act_given_sit", "low_hp", "attack_weakest"),
         ("act_given_sit", "in_range_cooling", "move_away"), ("act_given_sit", "enemy_far", "move_toward"),
         ("act_given_sit", "in_range_ready", "attack_weakest"), ("act", None, "move_away"),
         ("msg", None, "target_named"), ("msg", None, "kill"), ("msg", None, "backup"), ("msg", None, "retreat"),
         ("msg", None, "movement"), ("msg", None, "none")]


def get(t, k):
    a, b, c = k
    return t[a][b][c] if b else t[a][c]


def main():
    out = {"test": {}, "jsd": {}, "ci": {}, "train": {}, "eval": {}}
    games = {n: BS.load(s) for n, s in TEST}
    maps = sorted(set.intersection(*(set(g) for g in games.values())))
    per_map = {n: {m: counts([games[n][m][1]]) for m in maps} for n in games}
    for n in games:
        tot = collections.Counter()
        for m in maps:
            tot.update(per_map[n][m])
        out["test"][n] = table(tot)
    names = list(games)
    print(f"{len(maps)} shared test maps; decisions: " + ", ".join(f"{n} {out['test'][n]['n']}" for n in names))
    print("\n== situation mix (share of decisions)")
    print(f"{'':18s}" + "".join(f"{n:>11s}" for n in names))
    for s in SITS:
        print(f"{s:18s}" + "".join(f"{out['test'][n]['sit'][s]:11.3f}" for n in names))
    print("\n== P(action | situation)")
    for s in SITS:
        print(f"-- {s}")
        for a in ACTS:
            vals = [out["test"][n]["act_given_sit"][s][a] for n in names]
            if max(vals) >= 0.005:
                print(f"   {a:16s}" + "".join(f"{v:11.3f}" for v in vals))
    print("\n== message type, share of all decisions ('none' = no message)")
    for mc in MSG_CATS:
        print(f"{mc:18s}" + "".join(f"{out['test'][n]['msg'][mc]:11.3f}" for n in names))
    print(f"{'say-do':18s}" + "".join(f"{out['test'][n]['saydo']:11.3f}" for n in names))
    print("\n== Jensen-Shannon divergence (bits) between consecutive models")
    for a, b in zip(names, names[1:]):
        ja = jsd(out["test"][a]["act"], out["test"][b]["act"])
        js = sum(out["test"][a]["sit"][s] * jsd(out["test"][a]["act_given_sit"][s], out["test"][b]["act_given_sit"][s])
                 for s in SITS)
        jm = jsd(out["test"][a]["msg"], out["test"][b]["msg"])
        out["jsd"][f"{a}->{b}"] = dict(action=ja, action_within_situation=js, message=jm)
        print(f"  {a:>9s} -> {b:9s} action {ja:.3f}  action|situation {js:.3f}  message {jm:.3f}")
    print("\n== paired 95% bootstrap CI over maps of selected changes")
    cells = [(("sa", "low_hp", "move_away"), ("sit", "low_hp")), (("sa", "low_hp", "attack_weakest"), ("sit", "low_hp")),
             (("sa", "enemy_far", "move_toward"), ("sit", "enemy_far")),
             (("sa", "enemy_far", "move_away"), ("sit", "enemy_far")),
             (("sa", "in_range_cooling", "move_away"), ("sit", "in_range_cooling")),
             (("sa", "in_range_ready", "attack_weakest"), ("sit", "in_range_ready")),
             (("act", "invalid"), "n"), (("msg", "target_named"), "n"), (("msg", "kill"), "n"),
             (("msg", "movement"), "n"), (("msg", "backup"), "n"), (("msg", "retreat"), "n"), ("saydo", "named")]
    for a, b in zip(names, names[1:]):
        out["ci"][f"{a}->{b}"] = {}
        print(f"-- {a} -> {b}")
        for num, den in cells:
            lo, hi = boot_cell(per_map[a], per_map[b], maps, num, den)
            ra = sum(per_map[a][m][num] for m in maps) / max(sum(per_map[a][m][den] for m in maps), 1)
            rb = sum(per_map[b][m][num] for m in maps) / max(sum(per_map[b][m][den] for m in maps), 1)
            key = "/".join(x for x in (num if isinstance(num, tuple) else (num,)))
            out["ci"][f"{a}->{b}"][key] = dict(a=ra, b=rb, lo=lo, hi=hi)
            print(f"   {key:32s} {ra:.3f} -> {rb:.3f}  [{lo:+.3f}, {hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")

    # MAPPO training trajectory
    for label, path in [("train", "runs/mappo_full/episodes.jsonl"), ("eval", "runs/mappo_full/eval_episodes.jsonl")]:
        by = load_mappo(path)
        for it in sorted(by):
            t = table(counts([g for g, _ in by[it]]))
            t["win"] = sum(w for _, w in by[it]) / len(by[it])
            out[label][it] = t
    print("\n== MAPPO training rounds (train games: 16/round, T=1.0, random maps). Columns: rounds 0-22; "
          "slope per 10 rounds with 95% bootstrap CI")
    its = sorted(out["train"])
    for k in [("win", None, None)] + TRACK:
        ys = [out["train"][i]["win"] if k[0] == "win" else get(out["train"][i], k) for i in its]
        b, lo, hi = slope(its, ys)
        name = k[0] if k[0] == "win" else "/".join(x for x in k if x)
        out.setdefault("slopes", {})[name] = dict(per10=10 * b, lo=10 * lo, hi=10 * hi,
                                                  first5=sum(ys[:5]) / 5, last5=sum(ys[-5:]) / 5)
        print(f"  {name:40s} first5 {sum(ys[:5]) / 5:.3f}  last5 {sum(ys[-5:]) / 5:.3f}  slope/10 {10 * b:+.3f} "
              f"[{10 * lo:+.3f}, {10 * hi:+.3f}] {'*' if lo > 0 or hi < 0 else ''}")
    if "--json" in sys.argv:
        json.dump(out, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
    if "--fig-dir" in sys.argv:
        figs(out, Path(sys.argv[sys.argv.index("--fig-dir") + 1]))


def figs(out, d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d.mkdir(parents=True, exist_ok=True)
    names = list(out["test"])
    # (a) action given situation, stacked, for the three contested situations
    COL = {"attack_weakest": "#1F3A5F", "attack_other": "#6F8FB5", "move_toward": "#4E9A6B", "move_away": "#C0392B",
           "move": "#BBBBBB", "stop": "#888888", "invalid": "#F0B429"}
    sits = ["enemy_far", "in_range_cooling", "low_hp"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, s in zip(axes, sits):
        bottom = [0] * len(names)
        for a in ACTS:
            v = [100 * out["test"][n]["act_given_sit"][s][a] for n in names]
            ax.bar(names, v, bottom=bottom, color=COL[a], label=a.replace("_", " "), width=0.7)
            bottom = [b + x for b, x in zip(bottom, v)]
        ax.set_title(s.replace("_", " "), fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("% of decisions", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=7, fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, -0.04))
    plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(d / "fig_actions.pdf", bbox_inches="tight"); plt.close()
    # (b) message types among messages + no-message share
    MC = {"target_named": "#1F3A5F", "target_type": "#6F8FB5", "kill": "#4E9A6B", "backup": "#8E6FB5",
          "regroup": "#B58E6F", "retreat": "#C0392B", "kiting": "#E07B39", "movement": "#BBBBBB", "other": "#888888"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    bottom = [0] * len(names)
    for mc in MC:
        v = [100 * out["test"][n]["msg"][mc] for n in names]
        ax.bar(names, v, bottom=bottom, color=MC[mc], label=mc.replace("_", " "), width=0.7)
        bottom = [b + x for b, x in zip(bottom, v)]
    ax.set_ylabel("% of decisions with this message", fontsize=8); ax.tick_params(labelsize=7.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=6.3, frameon=False, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    plt.tight_layout(); plt.savefig(d / "fig_messages.pdf", bbox_inches="tight"); plt.close()
    # (c) MAPPO trajectory
    its = sorted(out["train"], key=int)
    sm = lambda ys, w=3: [sum(ys[max(0, i - w + 1):i + 1]) / len(ys[max(0, i - w + 1):i + 1]) for i in range(len(ys))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.3))
    for k, lab, c in [(("act_given_sit", "low_hp", "move_away"), "low-hp: move away", "#C0392B"),
                      (("act_given_sit", "low_hp", "attack_weakest"), "low-hp: attack weakest", "#1F3A5F"),
                      (("act_given_sit", "in_range_cooling", "move_away"), "cooling in range: move away", "#E07B39")]:
        ys = [100 * get(out["train"][i], k) for i in its]
        axes[0].plot([int(i) for i in its], ys, color=c, alpha=0.25, lw=0.8)
        axes[0].plot([int(i) for i in its], sm(ys), color=c, lw=1.6, label=lab)
    wins = [100 * out["train"][i]["win"] for i in its]
    axes[0].plot([int(i) for i in its], sm(wins), color="black", lw=1.2, ls="--", label="win rate (train games)")
    axes[0].set_xlabel("MAPPO round", fontsize=8); axes[0].set_ylabel("%", fontsize=8)
    axes[0].legend(fontsize=6.3, frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.3)); axes[0].tick_params(labelsize=7.5)
    for k, lab, c in [(("msg", None, "target_named"), "named target", "#1F3A5F"), (("msg", None, "kill"), "kill", "#4E9A6B"),
                      (("msg", None, "backup"), "backup", "#8E6FB5"), (("msg", None, "movement"), "movement", "#888888"),
                      (("msg", None, "none"), "no message", "#BBBBBB")]:
        ys = [100 * get(out["train"][i], k) for i in its]
        axes[1].plot([int(i) for i in its], ys, color=c, alpha=0.25, lw=0.8)
        axes[1].plot([int(i) for i in its], sm(ys), color=c, lw=1.6, label=lab)
    axes[1].set_xlabel("MAPPO round", fontsize=8); axes[1].set_ylabel("% of decisions", fontsize=8)
    axes[1].legend(fontsize=6.3, frameon=False, loc="center right"); axes[1].tick_params(labelsize=7.5)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    plt.tight_layout(); plt.savefig(d / "fig_mappo_trajectory.pdf", bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    main()
