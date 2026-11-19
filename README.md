# Small language models as a StarCraft II squad

Each unit of a SMACv2 squad is controlled by its own call to a small language model (Gemma 4 E4B).
We post-train that model in two ways and compare them:

1. **Learn from a teacher**: games played by an open-weight frontier model through OpenRouter
   (Gemma 4 31B), used for filtered behaviour cloning or advantage-weighted regression.
2. **Learn from itself**: online self-play RL (PPO + KL to the base model) against the
   StarCraft II built-in AI, with the environment running live during training.

A planned third experiment gives one unit a secret incentive and measures collusion / free-riding.


## Current status (2026-11-18)

- **Report draft**: `report/main.tex` / `report/main.pdf` (ICLR 2026 template, CS221 progress report).
- **Results so far** (64 fixed scenarios vs SC2 built-in AI, difficulty 7): untrained E4B 17.2%, DAgger from 31B 37.5%,
  reward-filtered SFT (AWR) from 31B 53.1%; Gemma 4 31B C0/C1/C2 = 25/50/60% (40 games each).
- **Multi-agent RL from the 53% SFT checkpoint (in progress)**: `smac_llm/mappo.py`
  - `--algo mappo`: shared LoRA actor, centralized critic on the SMACv2 global state, per-agent GAE, token-level PPO with
    sampled old log-probs, KL to the frozen SFT adapter.
  - `--algo grpo`: self-play GRPO (both teams share the policy, group = rollouts of the same scenario).
  - Reward: `ShapedReward2` in `smac_llm/game.py` (team reward + per-unit credit that sums to 0; unit test in `tests/`).
  - Behaviour analysis: `scripts/behavior_stats.py`.
- **Protocol**: `docs/unified_protocol.md`. **Related papers**: `paper/README.md`.
- Not in the repo: run outputs, logs, model adapters, videos, StarCraft II, API keys.

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
