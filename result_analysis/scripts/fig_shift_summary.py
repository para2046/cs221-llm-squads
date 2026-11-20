"""Figure: (a) how much behaviour changed between consecutive models (JSD of situation-conditioned actions vs of
message types, 64 test maps); (b, c) MAPPO checkpoint evaluations every 2 rounds on the same 16 dev maps (T=0.7):
message types and action rates / win rate.
Inputs: report/analysis/action_msg_shift.json, report/analysis/behavior_robustness.json.
usage: .venv-smac/bin/python scripts/fig_shift_summary.py [out.pdf]
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "report/fig_shift_summary.pdf"
J = json.load(open("report/analysis/action_msg_shift.json"))["jsd"]
E = json.load(open("report/analysis/behavior_robustness.json"))["eval_traj"]
its = list(range(0, 2 * len(E["win"]["values"]), 2))
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
pairs = [("untrained->SFT", "untr.\n→SFT"), ("SFT->MAPPO", "SFT→\nMAPPO"), ("MAPPO->31B", "MAPPO\n→31B")]
x = range(len(pairs))
ax[0].bar([i - 0.19 for i in x], [J[p]["action_within_situation"] for p, _ in pairs], 0.38, color="#1F3A5F",
          label="actions (within situation)")
ax[0].bar([i + 0.19 for i in x], [J[p]["message"] for p, _ in pairs], 0.38, color="#4E9A6B", label="message types")
for i, (p, _) in enumerate(pairs):
    a_, m_ = J[p]["action_within_situation"], J[p]["message"]
    for dx, v in ((-0.19, a_), (0.19, m_)):
        ax[0].text(i + dx, v + 0.01, f"{v:.3f}", ha="center", va="bottom", rotation=90, fontsize=6.5)
ax[0].set_xticks(list(x), [l for _, l in pairs], fontsize=8)
ax[0].set_ylim(0, 0.5); ax[0].set_ylabel("JSD (bits)", fontsize=9); ax[0].set_title("(a) change between models", fontsize=9)
ax[0].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
for k, lab, c in [("msg/none", "no message", "#9A9A9A"), ("msg/kill", "finish-off call", "#4E9A6B"),
                  ("msg/target_named", "names a target", "#1F3A5F")]:
    ax[1].plot(its, [100 * v for v in E[k]["values"]], "-o", ms=2.5, lw=1.3, color=c, label=lab)
ax[1].set_title("(b) checkpoints: messages", fontsize=9); ax[1].set_ylabel("% of decisions", fontsize=9)
ax[1].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
for k, lab, c, ls in [("enemy_far/move_toward", "out of range: move toward", "#1F3A5F", "-"),
                      ("low_hp/attack_weakest", "low hp: attack weakest", "#6F8FB5", "-"),
                      ("low_hp/move_away", "low hp: move away", "#C0392B", "-"),
                      ("win", "win rate (16 maps)", "black", "--")]:
    ax[2].plot(its, [100 * v for v in E[k]["values"]], ls, marker="o", ms=2.5, lw=1.2, color=c, label=lab)
ax[2].set_title("(c) checkpoints: actions, win", fontsize=9); ax[2].set_ylabel("%", fontsize=9)
ax[2].legend(fontsize=7.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
for a in ax[1:]:
    a.set_xlabel("MAPPO round (0 = SFT)", fontsize=8.5)
for a in ax:
    a.tick_params(labelsize=8)
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(w_pad=0.8)
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
