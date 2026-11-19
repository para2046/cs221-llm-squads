"""Render a few moments of one logged game as a static figure (for the report).

usage: python scripts/plot_frames.py runs/base_gemma4b/w2.jsonl --episode 0 --steps 0 12 30 \
           --out report/fig_frames.pdf
"""
import argparse
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MOVES = {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}
MARK = {"stalker": "D", "zealot": "o", "colossus": "h"}
ALLY, ENEMY = "#1f6fd1", "#c2421f"

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--episode", type=int, default=0)
p.add_argument("--steps", type=int, nargs="+", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()

frames = {}
for line in open(a.log):
    d = json.loads(line)
    if d.get("episode") == a.episode and not d.get("summary"):
        frames[d["step"]] = d
pts = [(u["x"], u["y"]) for f in frames.values() for u in f["units"]["allies"] + f["units"]["enemies"] if u["hp"] > 0]
x0, x1 = min(p[0] for p in pts) - 2, max(p[0] for p in pts) + 2
y0, y1 = min(p[1] for p in pts) - 2, max(p[1] for p in pts) + 2

fig, axes = plt.subplots(1, len(a.steps), figsize=(3.2 * len(a.steps), 3.3))
for ax, s in zip(axes, a.steps):
    f = frames[s]
    enemies = {e["id"]: e for e in f["units"]["enemies"]}
    for r in f["agents"]:
        me = next(u for u in f["units"]["allies"] if u["id"] == r["agent"])
        act = r["action"]
        if act >= 6 and (act - 6) in enemies:
            e = enemies[act - 6]
            ax.plot([me["x"], e["x"]], [me["y"], e["y"]], color=ALLY, lw=1, alpha=.7, zorder=1)
        elif act in MOVES:
            dx, dy = MOVES[act]
            ax.annotate("", xy=(me["x"] + 1.5 * dx, me["y"] + 1.5 * dy), xytext=(me["x"], me["y"]),
                        arrowprops=dict(arrowstyle="->", color=ALLY, lw=1.2), zorder=1)
    for side, color, prefix in (("allies", ALLY, "a"), ("enemies", ENEMY, "e")):
        for u in f["units"][side]:
            if u["hp"] <= 0:
                ax.scatter(u["x"], u["y"], marker="x", color="#888", s=40, zorder=2)
                continue
            frac = (u["hp"] + u["sh"]) / (u["hp_max"] + u["sh_max"])
            ax.scatter(u["x"], u["y"], marker=MARK.get(u["type"], "o"), color=color,
                       s=60 + 140 * (u["type"] == "colossus"), alpha=.35 + .65 * frac,
                       edgecolor="black", linewidth=.5, zorder=3)
            ax.text(u["x"], u["y"] + .9, f"{prefix}{u['id']}", ha="center", fontsize=7, zorder=4)
    alive_a = sum(u["hp"] > 0 for u in f["units"]["allies"])
    alive_e = sum(u["hp"] > 0 for u in f["units"]["enemies"])
    ax.set_title(f"step {s}: {alive_a} vs {alive_e} alive", fontsize=9)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
plt.tight_layout()
plt.savefig(a.out, bbox_inches="tight")
plt.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=160, bbox_inches="tight")
print("wrote", a.out)
