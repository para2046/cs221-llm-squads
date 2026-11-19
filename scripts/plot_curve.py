"""Plot an online RL learning curve (train_log.jsonl) against the untrained baseline.

usage: python scripts/plot_curve.py runs/rl_selfplay_v1/train_log.jsonl --baseline runs/base_gemma4b \
           --out report/fig_curve.pdf
"""
import argparse
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("log")
p.add_argument("--baseline", default=None, help="dir of baseline episode JSONL files")
p.add_argument("--window", type=int, default=5)
p.add_argument("--split", type=float, default=None, help="draw a marker where the batch size changed")
p.add_argument("--out", required=True)
a = p.parse_args()

rows = [json.loads(l) for l in open(a.log)]
it = [r["iter"] for r in rows]


def rolling(xs, w):
    return [sum(xs[max(0, i - w + 1):i + 1]) / len(xs[max(0, i - w + 1):i + 1]) for i in range(len(xs))]


base = None
if a.baseline:
    eps = [json.loads(l) for f in glob.glob(f"{a.baseline}/*.jsonl") for l in open(f)]
    eps = [e for e in eps if e.get("summary")]
    n = len(eps)
    mr = sum(e["return"] for e in eps) / n
    sd = (sum((e["return"] - mr) ** 2 for e in eps) / (n - 1)) ** 0.5
    base = {"ret": mr, "ret_ci": 1.96 * sd / n ** 0.5,
            "win": sum(e["won"] for e in eps) / n, "n": n}

fig, axes = plt.subplots(1, 3, figsize=(11, 2.8))
for ax, key, label in ((axes[0], "return", "team return"), (axes[1], "win_rate", "win rate"),
                       (axes[2], "kl", "KL to base model")):
    ys = [r[key] for r in rows]
    ax.plot(it, ys, color="#9db8d9", lw=1, marker="o", ms=2.5, label="per iteration")
    ax.plot(it, rolling(ys, a.window), color="#1f6fd1", lw=2, label=f"{a.window}-iteration mean")
    if base and key == "return":
        ax.axhspan(base["ret"] - base["ret_ci"], base["ret"] + base["ret_ci"], color="#c2421f", alpha=.12)
        ax.axhline(base["ret"], color="#c2421f", lw=1, ls="--", label=f"untrained ({base['n']} games)")
    if base and key == "win_rate":
        ax.axhline(base["win"], color="#c2421f", lw=1, ls="--")
    if a.split is not None:
        ax.axvline(a.split - .5, color="#888", lw=1, ls=":")
    ax.set_xlabel("iteration")
    ax.set_title(label, fontsize=10)
    ax.grid(alpha=.3)
axes[0].legend(fontsize=7, loc="lower right")
plt.tight_layout()
plt.savefig(a.out, bbox_inches="tight")
plt.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=160, bbox_inches="tight")
print("wrote", a.out, "iterations:", len(rows))
