# Result analysis: where do the gains come from?

Mentor question (2026-11-19 meeting): from untrained Gemma 4 E4B to E4B after SFT on 31B games, where does the
improvement come from? And from SFT to MAPPO?

This folder has the raw game logs used, the analysis outputs, the scripts, and a summary of the findings so far.
Everything is computed from the logged prompts (what each unit saw) and replies (what it ordered); nothing is
re-simulated.

## Setting (short)

- SMACv2 `10gen_protoss`, 5 vs 5, vs the StarCraft II built-in AI at difficulty 7. One LLM call per unit per step.
  Channel C2: teammates' states plus a free message of at most 8 words. Each unit also sees its own last 5 steps
  (the "history").
- The seed (scenario id) fixes the unit mix (mirrored for the enemy) and the start type: grouped (our 5 units around
  the map centre) or split. Maps are paired by seed across models.
- 4B evaluation sampling: temperature 0.7, top-k 64 for every 4B run. 31B on OpenRouter: temperature 0.7.
- Map sets:
  - **dev**: seeds 100000-100063. Untrained and SFT exist here; this set was also used during development.
  - **new**: seeds 200000-200063, never used for training or model selection. SFT, MAPPO, self-play GRPO and 31B
    exist here. The untrained 4B with and without history now also exists here (added 2026-11-19 evening), so all four models
    (untrained, SFT, MAPPO, 31B) share one map set: see `outputs/behavior_shift_new_*.txt` and
    `outputs/gain_new_3models.txt`. The findings below were first computed with untrained on the dev maps; on the
    new maps the same pattern holds (untrained 21.9% → SFT 53.1%, maps only one won 5 vs 25; target switch 13.8% → 2.1%,
    moves away 28.7% → 6.2%, invalid 5.4% → 0.2%, all already in the opening).
  - History effect on the new maps is much smaller than on dev: 21.9% vs 15.6% (9 vs 5, p = 0.42); dev was 17.2% vs 3.1%.

## Logs (`logs/`)

On GitHub the logs are gzipped (`*.jsonl.gz`); run `gunzip -k result_analysis/logs/*.gz result_analysis/logs/new_31b/*.gz` first.

One JSON line per game. In the 4B files, `decisions` holds one entry per unit per step
(`t`, `agent`, `prompt`, `reply`, `valid`), and `info` has `battle_won`, `dead_allies` and `dead_enemies`.
The 31B files are `play.py` logs: one line per step with `team_a` records, and a `summary` line per game
(`scenario`, `won`).

| File | Model | Maps | Win |
|---|---|---|---|
| `dev_untrained_4b.jsonl` | E4B, untrained, C2 + history | dev | 17.2% (11/64) |
| `dev_untrained_4b_nohistory.jsonl` | E4B, untrained, C2, no 5-step history | dev | 3.1% |
| `dev_sft_4b.jsonl` | E4B after SFT on 31B games | dev | 53.1% |
| `new_sft_4b.jsonl` | same SFT model | new | 53.1% (34/64) |
| `new_mappo_4b.jsonl` | SFT + MAPPO, 23 rounds (best RL so far) | new | 60.9% (39/64) |
| `new_grpo_selfplay_4b.jsonl` | SFT + self-play GRPO, 30 rounds (dropped) | new | 43.8% |
| `new_31b/` | Gemma 4 31B (teacher), via OpenRouter, eval only | new | 70.3% (45/64) |
| `new_untrained_4b.jsonl` | E4B, untrained, C2 + history | new | 21.9% (14/64) |
| `new_untrained_4b_nohistory.jsonl` | E4B, untrained, no history | new | 15.6% (10/64); vs history: maps only one won 5 vs 9, p = 0.42 |

Use `team == "A"` only. In the 4B files, team A is ours and team B is absent (built-in AI).

## Scripts (`scripts/`; the originals live in `../scripts/`)

- `gain_analysis.py`: per-game outcome and timing features (who loses a unit first, wins after losing the first
  unit) and per-decision features split by phase (even / behind / ahead in alive units). Paired bootstrap CIs.
- `behavior_shift.py`: per-game behaviour features (reuses `win_loss_analysis.features`), paired change between
  models with 95% bootstrap CIs over maps, and an optional phase restriction:
  - `--phase opening`: only steps before either side loses a unit, so the feature is less driven by already
    winning or losing.
  - `--phase after`: the rest of the game.
- `win_loss_analysis.py`: the earlier analysis of what separates won from lost games over 728 games of many players
  (logit with one intercept per player). Its output is `outputs/win_loss_features_728games.json`.
- `same_maps_table.py`: win rates on the new maps (grouped vs split), giving `outputs/same_maps.json`.
- `msg_taxonomy.py` and `behavior_stats.py`: older helpers (message categories, simple behaviour rates).

Run from the project root, e.g.
`python scripts/behavior_shift.py --maps new --phase opening`. The paths in `MODELS` point at `runs/...`; the
same files are copied in `logs/` here.

## Outputs (`outputs/`)

- `behavior_shift_{dev,new}_{all,opening,after}.txt|json`: feature means per model, paired changes with CIs, and the
  win-model decomposition (see the caveat below).
- `gain_untrained_sft_dev.json` and `gain_sft_mappo_f64.json`: the `gain_analysis.py` results.

## Feature definitions

| Feature | Definition |
|---|---|
| `attack` | share of decisions that are an attack order |
| `focus` | steps with at least 2 attackers where all of them hit one target |
| `targets_per_step` | distinct targets per step with at least 2 attackers |
| `weakest` | attack on the lowest hp+shield enemy in range |
| `target_switch` | left a target that is still in range for another one |
| `ready_noshoot` | weapon ready and enemy in range, but no attack |
| `cd_ir_move` | weapon cooling and enemy in range: moved |
| `low_hp_move` | below 35% hp+shield and enemy in range: moved |
| `low_hp_away` | below 35% hp+shield, enemy visible, moved away from the nearest enemy |
| `move_away` | enemy visible, a move order that does not go toward the nearest enemy |
| `stalker_kite` | stalker reloading with a zealot within 4: moved |
| `sync_at_contact` | share of our units near an enemy at first contact |
| `cohesion_pre` / `cohesion_fight` | mean distance of our units to their centroid before / after contact (lower = tighter) |
| `regroup` | spread at spawn minus spread just before contact (positive = gathered) |
| `alone_with_enemy` | sees an enemy with no teammate in sight |
| `invalid` | illegal or unparseable reply |
| `msg_rate` | share of decisions with a message |
| `msg_named` | share of decisions with a message that names an enemy id (e.g. "e3") |

## Findings so far

### 1. Untrained 4B → SFT on 31B games (dev maps, 64 paired)

Win rate 17.2% → 53.1% (paired difference +36 points, 95% CI +22 to +50).

- **The opening is not where it wins.** Who loses a unit first is about the same: the untrained model loses first in
  69% of games, SFT in 58% (CI includes 0).
- **It wins the rest of the game.** After losing the first unit, wins go from 2% to 38%. After killing first,
  56% → 77%.
- **Kills and losses per game:** 2.8 → 4.1 enemies killed, 4.7 → 3.7 own units lost.

Behaviour changes. All of these have CIs that exclude 0, and most are already present in the opening, before any
unit is lost.

| Behaviour | Untrained | SFT |
|---|---|---|
| Target switch (target still in range) | 14% | 2% |
| Moves away with enemies visible | 27% | 5% |
| Spread during the fight (lower = tighter) | 4.4 | 3.4 |
| Regroups before contact | no (−0.1) | yes (+0.1) |
| Invalid orders | 5.2% | 0.1% |
| Attacks the weakest enemy in range | 84% | 90% |
| Sees an enemy with no teammate in sight | 15% | 8% |
| Decisions with a message that names a target | 8% | 68% |

**Interpretation:** the untrained model wavers. It switches targets, walks away and breaks formation, so it is
picked off one unit at a time. SFT copies 31B's core pattern: stay together, keep hitting one target, use valid
orders, and call targets in messages.

### 2. SFT → MAPPO (new maps, 64 paired)

Win rate 53.1% → 60.9% (+8 points, 95% CI −9 to +27, **not yet significant**).

- **The opening is unchanged.** The model that loses a unit first is the same in about 47% vs 48% of games.
- **The whole gain is in fights after losing the first unit:** 23% → 52% wins.

Behaviour changes are small, and all point the same way: **it retreats less**.

| Behaviour | SFT | MAPPO |
|---|---|---|
| Low-HP units moving away (CI excludes 0) | 26% | 14% |
| Low-HP units moving away while behind | 64% | 15% |
| Stalker kiting (CI excludes 0) | 5% | 2% |
| Moves away after the first loss (CI excludes 0) | | 7 points lower |
| All attackers on one target | 34% | 41% (borderline) |
| Same, when ahead | 55% | 68% |
| Messages | 81% | 86% |

**Interpretation:** when SFT falls behind, damaged units back off, which removes damage from the fight. MAPPO keeps
them fighting and focuses fire.

### 3. Contrast with the 31B teacher (new maps)

31B (70%) does the opposite of MAPPO.

| Behaviour | SFT | MAPPO | 31B |
|---|---|---|---|
| Stalker kiting | 5% | 2% | 21% |
| Low-HP units moving away | 26% | 14% | 43% |
| Moves away with enemies visible | 6% | 4% | 15% |
| Weakest target | 90% | 90% | 93% |
| Share of units engaged at first contact | 0.50 | 0.51 | 0.37 |

- **SFT copied:** 31B's targeting and messages.
- **SFT did not copy:** the rarer repositioning moves (kiting, backing off), which imitation learns poorly.
- **MAPPO** moved further from the teacher, toward a simpler "stay and fight" strategy that works for the 4B.

## Caveats and open questions for the reviewer

- **These are correlations, not causal effects.** Phase splits ("behind", "after the first loss") are selected by
  the outcome itself. The `opening` phase reduces this, but does not remove it.
- **The decomposition is not reliable.** It predicts log-odds change as β × feature change, with β from a pooled
  logit over many players. The behaviour features are collinear (e.g. `alone_with_enemy` and `cohesion_fight`), and
  for untrained → SFT it explains more than the actual change (+3.9 vs +1.7 log-odds). We do not plan to report it
  as a result.
- **Done:** the untrained → SFT comparison has been redone on the new maps (the report uses those numbers).
- **Planned causal checks** (need GPU time; not run yet):
  - SFT with teammates' messages removed (`env_server --msg-ablation drop`) on the 64 new maps, to measure how much
    of the gain comes from communication.
  - The same for MAPPO.
- **Other possible checks:**
  - per unit type (zealot, stalker, colossus) and per start type (grouped vs split; the gaps between models are
    mainly on split maps);
  - say–do consistency (a unit that names a target attacks it);
  - per-game style clusters.

## Action and message shift, and the MAPPO training trajectory (`scripts/action_msg_shift.py`)

Outputs: `outputs/action_msg_shift.txt|json`, figures `outputs/fig_actions.pdf`, `fig_messages.pdf`,
`fig_mappo_trajectory.pdf`. Training logs: `logs/mappo_training/`. These hold all 23 rounds of MAPPO training games
(16 per round, T=1.0, random maps) and the evaluation every 2 rounds (16 fixed dev maps, T=0.7). Eval round 0 is the
SFT policy.

Every decision is put into a **situation**: no enemy / enemy visible but none in range / in range with the weapon
ready / in range while reloading / low HP (<35%) with an enemy visible. Within each situation we take the
**action class**: attack weakest, attack other, move toward, move away, move (no enemy), stop, invalid.
Messages get one **type** each, using the same regexes as `msg_taxonomy.py`. **Say–do** is the share of messages
naming an enemy where the unit itself attacks that enemy in the same reply.

Key numbers (64 test maps):

| | untrained | SFT | MAPPO | 31B |
|---|---|---|---|---|
| Enemy far, moves toward it | 54% | 97% | 96% | 97% |
| Enemy far, moves away | 20% | 2.5% | 4.0% | 2.0% |
| Enemy far, invalid | 26% | 0.1% | 0% | 0.8% |
| No enemy, stops | 32% | 0.9% | 0.1% | 0.2% |
| Low HP, attacks weakest | 65% | 80% | 82% | 74% |
| Low HP, moves away | 9.0% | 2.2% | 1.4% | 7.4% |
| No message | 77% | 19% | 14% | 14% |
| Message names a target | 8% | 63% | 64% | 60% |
| Kill confirmation | 0.2% | 8.8% | 12.7% | 11.5% |
| Say–do | 12% | 80% | 79% | 77% |
| Action JSD from the previous model, within situation | – | 0.072 bits | 0.002 | 0.014 |
| Message JSD from the previous model | – | 0.375 bits | 0.009 | 0.016 |

**Correction to the earlier "MAPPO retreats less" finding.** `low_hp_away` in `behavior_shift` is the share of
**moves** that go away. Low-HP units rarely move: 80% of their decisions are attacks. As a share of all low-HP
decisions, moving away is 2.2% (SFT) vs 1.4% (MAPPO), and the CI includes 0. MAPPO's action distribution is almost
identical to SFT's (JSD 0.002 bits).

**MAPPO training trajectory** (slope per 10 rounds, 95% bootstrap CI):

- Messages change steadily:
  - no-message share 19% → 12% (−3.7 points per 10 rounds, significant);
  - kill confirmations 9% → 13% (+2.1, significant);
  - named targets +1.6 (significant).
- Action rates within each situation show no significant trend.
- The training-game win rate shows no trend (16 games per round, ±12 points of noise).

## Robustness checks (Nov 19, after reviewing a second agent's handoff)

Script: `scripts/behavior_robustness.py` → `outputs/behavior_robustness.{txt,json}`, `outputs/msg_audit_sample.txt`.

- **Per-game averages vs pooled decisions:** every Table 2 row agrees within 1–2 points. With per-game averages no SFT→MAPPO action change is significant; the message changes are (no message −5.3 points, finish-off calls +4.1, mentions an enemy +5.2).
- **The "kill" category is mislabelled:** 96–100% of its messages are "finishing eX" calls (finish off a damaged enemy), not kill reports. It is now called "finish-off call" in the report and figures.
- **Audit (25 random messages per category):** labels are accurate, except *retreat* for 31B, where 78% are real retreats (others are e.g. "low hp, attacking e1").
- **Checkpoint evals during MAPPO** (the same 16 dev maps every 2 rounds, T=0.7) repeat the training-game trends: no message −3.7 points per 10 rounds (−6.8 to −0.9), finish-off calls +2.4 (+1.9 to +3.0), no action trend, flat win rate.
- **The second agent's draft LaTeX** used the old share-of-moves "low-hp retreat 25.8→13.7%" and "kite" numbers; those are superseded and were not used.
- **To do later:** evaluate MAPPO checkpoints (adapter_it4/8/12/16/20) on the 64 test maps.
