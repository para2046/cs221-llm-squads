"""Fig. 1 of the progress report: win rate vs built-in AI under the unified protocol (95% Wilson CIs)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z*z/n; c = (p + z*z/(2*n)) / d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, c - h, c + h
groups = [("Gemma 4 31B (40)", [("C0", 10, 40), ("C1", 20, 40), ("C2", 24, 40)]),
          ("Gemma 4 E4B (64)", [("C0", 11, 64), ("C1", 12, 64), ("C2", 11, 64), ("C2\nno hist.", 2, 64)]),
          ("E4B self-play\nRL, C2 (64)", [("v1", 15, 64), ("v2", 9, 64), ("v2.1", 13, 64)]),
          ("E4B learns\nfrom 31B, C2 (64)", [("AWR", 34, 64), ("DAgger", 24, 64)]),
          ("Rule bots\n(40)", [("focus", 20, 40), ("kite", 21, 40)])]
cols = ["#4C72B0", "#55A868", "#C44E52", "#DD8452", "#8172B2"]
fig, ax = plt.subplots(figsize=(7.2, 2.1)); x = 0; ticks = []; labs = []
for gi, (g, cells) in enumerate(groups):
    xs = []
    for lab, k, n in cells:
        p, lo, hi = wilson(k, n)
        ax.bar(x, 100*p, color=cols[gi], width=0.8)
        ax.errorbar(x, 100*p, yerr=[[100*(p-lo)], [100*(hi-p)]], color="k", capsize=2, lw=0.8)
        ticks.append(x); labs.append(lab); xs.append(x); x += 1
    ax.text(np.mean(xs), 84, g, ha="center", fontsize=6.5); x += 0.7
ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=7); ax.set_ylabel("win rate (%)", fontsize=8)
ax.set_ylim(0, 95); ax.tick_params(axis="y", labelsize=7); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("report/fig_winrate.pdf"); plt.savefig("report/fig_winrate.png", dpi=200)
