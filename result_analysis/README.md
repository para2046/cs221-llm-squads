# Game logs behind the progress report

The raw game logs used for the win rates and the behaviour analysis in the CS221 progress report (Q2–Q5; the report
itself is not in this repository). The analysis scripts are in `../scripts/`; the commands below write their outputs
to a local `report/` folder.

## Setting

- SMACv2 `10gen_protoss`, 5 vs 5, against the StarCraft II built-in AI at difficulty 7. One LLM call per unit per step.
  Channel C2: teammates' states plus a free message of at most 8 words; each unit also sees its own last 5 steps.
- Maps are paired by seed across models.
  - **dev**: seeds 100000–100063, used during development.
  - **test**: seeds 200000–200063, never used for training or model selection.
- Sampling: the 4B uses temperature 0.7, top-k 64; 31B (on OpenRouter) uses temperature 0.7.

## Logs (`logs/`, gzipped)

Run `gunzip -k result_analysis/logs/*.gz result_analysis/logs/*/*.gz` first.
- 4B files have one JSON line per game:
  - `decisions` holds one entry per unit per step: `t`, `agent`, `prompt`, `reply`, `valid`.
  - `info` holds `battle_won`.
  - Use `team == "A"` only.
- The 31B files are `smac_llm/play.py` logs: one line per step with `team_a` records, plus a `summary` line per game.

| File | Model | Maps | Win | Report |
|---|---|---|---|---|
| `dev_untrained_4b.jsonl.gz` | E4B untrained, C2 + history | dev | 17.2% | Table 1, Q2 |
| `dev_untrained_4b_nohistory.jsonl.gz` | E4B untrained, no history | dev | 3.1% | Table 1, Q2 |
| `dev_sft_4b.jsonl.gz` | E4B after SFT on 31B games | dev | 53.1% | Q3 |
| `new_untrained_4b.jsonl.gz` | E4B untrained | test | 21.9% | Fig. 1, Q2, Q5 |
| `new_untrained_4b_nohistory.jsonl.gz` | E4B untrained, no history | test | 15.6% | Fig. 1, Q2 |
| `new_sft_4b.jsonl.gz` | SFT | test | 53.1% | Fig. 1, Q3–Q5 |
| `new_mappo_4b.jsonl.gz` | SFT + MAPPO, 23 rounds | test | 60.9% | Fig. 1, Q4–Q5 |
| `new_31b/` | Gemma 4 31B (teacher), evaluation only | test | 70.3% | Fig. 1, Q3, Q5 |
| `mappo_training/train_games_rounds0-22.jsonl.gz` | MAPPO training games (16 per round, T = 1.0, random maps) | – | – | Q5 slopes |
| `mappo_training/eval_every2_dev16.jsonl.gz` | MAPPO checkpoints every 2 rounds, the same 16 dev maps | dev | – | Fig. 2b–c |
| `mappo_training/train_log.jsonl.gz` | per-round training diagnostics (KL, clip fraction) | – | – | Q4 |

## Reproducing the report's analysis

```bash
python scripts/same_maps_table.py                   # Figure 1 win rates
python scripts/gain_analysis.py ...                 # who loses a unit first, wins after the first loss
python scripts/behavior_shift.py --maps new --phase opening   # behaviour in the opening only
.venv-smac/bin/python scripts/action_msg_shift.py --json report/analysis/action_msg_shift.json --fig-dir report
.venv-smac/bin/python scripts/behavior_robustness.py --json report/analysis/behavior_robustness.json \
    --audit report/analysis/msg_audit_sample.txt     # per-game averages, checkpoint evals, label audit
.venv-smac/bin/python scripts/fig_shift_summary.py  # Figure 2 bottom
```

The scripts read `runs/...`; these logs are the same files, gzipped.
Message categories are keyword labels (`scripts/msg_taxonomy.py`).
The category first called "kill" is almost entirely "finishing eX" calls, so the report calls it a finish-off call.
