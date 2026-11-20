"""Report figure: win rate of every model on the same 64 test maps, 95% Wilson intervals.
usage: python scripts/fig_same_maps.py report/slides_src/same_maps.json report/fig_same_maps.pdf"""
import json, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
ORDER = [("Gemma 4 31B (teacher)", "Gemma 4 31B (teacher)", "#1F3A5F"),
         ("GPT-5.6 Luna", "GPT-5.6 Luna (eval only)", "#8A939C"),
         ("Gemini 3.5 Flash-Lite", "Gemini 3.5 Flash-Lite (eval only)", "#8A939C"),
         ("4B MAPPO (best RL so far)", "E4B SFT + MAPPO", "#B5172F"),


         ("4B SFT on 31B games", "E4B SFT on 31B games", "#B5172F"),
         ("4B untrained", "E4B untrained", "#D9808C"),
         ("4B untrained, no history", "E4B untrained, no history", "#D9808C")]
rows = {r["model"]: r for r in d["rows"]}
items = [(lab, rows[k], c) for k, lab, c in ORDER if k in rows]


def wilson(k, n, z=1.96):
    p = k / n; den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den; half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (mid - half), 100 * (mid + half)


fig, ax = plt.subplots(figsize=(6.6, 0.33 * len(items) + 0.6))
for i, (lab, r, c) in enumerate(items[::-1]):
    lo, hi = wilson(r["wins"], r["n"])
    ax.barh(i, r["win"], color=c, height=0.62)
    ax.plot([lo, hi], [i, i], color="black", lw=0.9)
    ax.text(hi + 1.2, i, f"{r['win']:.1f}%  ({r['wins']}/{r['n']})", va="center", fontsize=8)
ax.set_yticks(range(len(items)))
ax.set_yticklabels([lab for lab, _, _ in items[::-1]], fontsize=8.5)
ax.set_xlim(0, 100); ax.set_xlabel("win rate vs built-in AI, difficulty 7 (%)", fontsize=8.5)
ax.tick_params(axis="x", labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="x", color="#dddddd", lw=0.5); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(sys.argv[2], bbox_inches="tight"); plt.savefig(sys.argv[2].replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
