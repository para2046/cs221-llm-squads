# Small language models as a StarCraft II squad

Each unit of a SMACv2 squad is controlled by its own call to a small language model (Gemma 4 E4B).
We post-train that model in two ways and compare them:

1. **Learn from a teacher (SFT on 31B games)**: games played by an open-weight model through OpenRouter
   (Gemma 4 31B); the 4B is fine-tuned to reproduce the 31B's replies, after dropping clearly worse decisions.
2. **Learn by playing (multi-agent RL)**: starting from that SFT model, MAPPO and GRPO against the
   StarCraft II built-in AI (and, in one GRPO variant, also against the 31B), with the environment running live.

A planned third experiment gives one unit a secret incentive and measures collusion / free-riding.


## Current status (2026-11-19)

- **Report draft**: `report/main.tex` / `report/main.pdf` (ICLR 2026 template, CS221 progress report).
- **Results so far** (vs SC2 built-in AI, difficulty 7):
  - Same 64 new maps for every model (seeds 200000–200063, never used in training): Gemma 4 31B with messages (C2)
    70.3%, GPT-5.6 Luna 64.1%, Gemini 3.5 Flash-Lite 53.1% (API models: evaluation only, never training data);
    E4B SFT on 31B games 53.1%; E4B MAPPO (best RL so far) 60.9%. On maps only one of them won, 31B beat SFT 17 to 6;
    MAPPO vs SFT is 20 to 15, a lead that is not yet confirmed.
  - Earlier tests: untrained E4B 17.2% (64 development maps, 100000+); 31B C0/C1/C2 = 25/50/60% (40 random games each).
- **Multi-agent RL from the 53% SFT checkpoint (in progress)**: `smac_llm/mappo.py`
  - `--algo mappo`: shared LoRA actor, centralized critic on the SMACv2 global state, per-agent GAE, token-level PPO with
    sampled old log-probs, KL to the frozen SFT adapter.
  - `--algo grpo`: critic-free GRPO (group = G games on the same map; advantage = team return relative to the group
    mean, plus each unit's own credit). All 5 units share one LoRA. The opponent depends on the env server:
    - **vs the built-in AI (no self-play)**: point `--env-url` at a server started with `--opponent builtin`
      (`scripts/grpo_ai_full.sh`).
    - **mixed opponents**: add `--opp-env-url` (a server with `--opponent external`) and `--opp-frac 0.5`; half of each
      round's groups are played against an API model (default Gemma 4 31B on OpenRouter, `--opp-llm`), the rest
      against the built-in AI. The API model's moves are never training data; a failed API call stops the run.
    - self-play: `--env-url` on a server with `--opponent external` and no `--opp-env-url` (both teams = the policy).
  - Same-map evaluation of API models: `smac_llm/play.py --scenario-start 200000` plays the fixed maps
    (`scripts/fixed40_api.sh`); `scripts/same_maps_table.py` builds the comparison on maps 200000–200063.
  - Reward: `ShapedReward2` in `smac_llm/game.py` (team reward + per-unit credit that sums to 0; unit test in `tests/`).
  - Behaviour analysis: `scripts/behavior_stats.py`.
- **Prompts and channel examples**: `docs/prompts.md`. **Protocol**: `docs/unified_protocol.md`. **Related papers**: `paper/README.md`.
- Not in the repo: run outputs, logs, model adapters, videos, StarCraft II, API keys.


## To-do (agreed next steps)

1. **Compare multi-agent RL methods on top of the 4B SFT model (in progress).** Start every method from the same
   53% SFT model with one shared policy for all units: MAPPO (centralized critic on the global state) vs GRPO against
   the built-in AI, and GRPO against built-in AI + 31B. Same budget; compare on the same 64 new maps (200000–200063).
   So far MAPPO (23 rounds) 60.9% vs SFT 53.1% (maps only one won: 20 to 15). Self-play GRPO was dropped
   (43.8%: it learned to beat itself, not the built-in AI).
2. **Memory of past games: access to the log of the previous round under the same seed.** Before a game, agents can
   read what happened the last time this scenario (seed) was played: their own actions, outcomes and messages,
   possibly as model-written, team-shared lessons (design: `docs/memory_design.md`).
   - This is cross-game, individual learning rather than coordination as such.
   - It must be tested on seeds never used for analysis or model selection, so it doesn't overfit the evaluation set.
3. **Collusion: self reward above team reward.** Give one or more units a secret individual incentive that conflicts
   with the team goal, then measure free-riding and collusion.
   - Measures: who takes damage, kill stealing, say–do consistency, whether messages reveal or hide the incentive.
   - Related: `paper/` (Secret Collusion, Colosseum) and `docs/coordination_lit.md`.

## Layout

```
smac_llm/text_env.py     text observation per unit, legal-action menu, reply parsing, fallback
smac_llm/run.py          play and log games with random / any OpenAI-compatible model
smac_llm/env_server.py   HTTP server running N StarCraft II games on a CPU node (text in, text out)
smac_llm/train.py        --mode bc | online | eval   (LoRA on Gemma 4, one GPU)
smac_llm/serve_gemma.sh  vLLM server for fast evaluation
scripts/collect.sh       parallel game collection inside the CPU Slurm job
scripts/summarize.py     win rate / return / kills / illegal-order rate with 95% CIs
scripts/build_replay.py  self-contained interactive replay page from a game log
scripts/plot_frames.py   static figure of a few moments of a game
report/                  progress report (ICLR 2026 template)
```

## Setup

```bash
# StarCraft II 4.10 (Linux, headless) + SMAC maps
export SC2PATH=$PWD/external/sc2/StarCraftII
uv venv .venv-smac --python 3.10 && uv pip install --python .venv-smac/bin/python -e external/smacv2 openai requests matplotlib
uv venv .venv-train --python 3.12 && uv pip install --python .venv-train/bin/python torch transformers peft accelerate requests
echo "OPENROUTER_API_KEY=..." > .env && chmod 600 .env
```

## Run

```bash
# play 5 games with an OpenRouter model (logged to JSONL, one line per step)
python -m smac_llm.run --map 10gen_protoss --n-units 5 --policy llm \
  --base-url https://openrouter.ai/api/v1 --model google/gemma-4-31b-it \
  --api-key-env OPENROUTER_API_KEY --episodes 5 --out runs/teacher/w0.jsonl

# environment server (CPU node) and online self-play RL (GPU node)
python -m smac_llm.env_server --n-envs 8 --port 8390 --token-file data/env_token
python -m smac_llm.train --mode online --env-url http://<cpu-node>:8390 --env-token-file data/env_token \
  --out runs/rl --iters 40 --episodes-per-iter 8

# imitation from teacher games, then evaluation
python -m smac_llm.train --mode bc --data runs/teacher --weighting awr --out runs/bc
python -m smac_llm.train --mode eval --adapter runs/bc/adapter --env-url ... --out runs/bc_eval --iters 3

# replay page
python scripts/build_replay.py runs/teacher/w0.jsonl --model "Gemma 4 31B" --out artifacts/replay/index.html
```

## Pilot results (SMACv2 10gen_protoss 5v5, built-in AI difficulty 7)

| Policy | Games | Win rate | Team return | Illegal orders |
|---|---|---|---|---|
| Random legal actions | 20 | 0% | 5.4 | 0% |
| Gemma 4 E4B, untrained | 20 | 20% | 12.5 | 8% |
| Gemma 4 31B (teacher) | 3 | 2/3 | 18.7 | 0.2% |
