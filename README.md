# Small language models as a StarCraft II squad

Each unit of a SMACv2 squad is controlled by its own call to a small language model (Gemma 4 E4B).
We post-train that model in two ways and compare them:

1. **Learn from a teacher (SFT on 31B games)**: games played by an open-weight model through OpenRouter
   (Gemma 4 31B); the 4B is fine-tuned to reproduce the 31B's replies, after dropping clearly worse decisions.
2. **Learn by playing (multi-agent RL)**: starting from that SFT model, MAPPO against the StarCraft II built-in AI,
   with the environment running live.

A planned third experiment gives one unit a secret incentive and measures collusion / free-riding.


## Demo

Real SC2 video with team messages: https://claude.ai/artifact/KBMsSf2gCHgWGRyRqchZzW

## Progress report (2026-11-20)

CS221 progress report (not included in this repository). Results, all against the SC2 built-in AI at
difficulty 7, on the same 64 test maps (seeds 200000–200063, never used in training):

| Model | Win |
|---|---|
| Gemma 4 31B (teacher), C2 | 70.3% |
| E4B SFT + MAPPO | 60.9% (vs SFT: maps only one won 20 to 15, not significant) |
| E4B SFT on 31B games | 53.1% |
| E4B untrained | 21.9% (no history: 15.6%) |

Closed models (GPT-5.6 Luna, Gemini 3.5 Flash-Lite) are evaluation only.
Game logs and how to reproduce the analysis: `result_analysis/README.md`.

Main code:
- `smac_llm/train.py`: SFT (reward-filtered BC on 31B games) and evaluation.
- `smac_llm/mappo.py`: MAPPO with a shared LoRA actor, a centralized critic, and a KL anchor to SFT.
- `smac_llm/game.py`: text protocol and shaped reward.
- `smac_llm/env_server.py`: the StarCraft II game server.

Behaviour analysis:
- `scripts/action_msg_shift.py`
- `scripts/behavior_robustness.py`
- `scripts/behavior_shift.py`
- `scripts/gain_analysis.py`

Next steps (see the report):
- Confirm the MAPPO gain on more maps and checkpoints.
- Message drop/shuffle tests.
- Memory of the previous game on the same seed.
- Secret individual incentives (collusion).

## Layout

```
smac_llm/text_env.py     text observation per unit, legal-action menu, reply parsing, fallback
smac_llm/run.py          play and log games with random / any OpenAI-compatible model
smac_llm/env_server.py   HTTP server running N StarCraft II games on a CPU node (text in, text out)
smac_llm/train.py        --mode bc | online | eval   (LoRA on Gemma 4, one GPU)
smac_llm/serve_gemma.sh  vLLM server for fast evaluation
scripts/collect.sh       parallel game collection inside the CPU Slurm job
scripts/summarize.py     win rate / return / kills / illegal-order rate with 95% CIs
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
```

## Pilot results (SMACv2 10gen_protoss 5v5, built-in AI difficulty 7)

| Policy | Games | Win rate | Team return | Illegal orders |
|---|---|---|---|---|
| Random legal actions | 20 | 0% | 5.4 | 0% |
| Gemma 4 E4B, untrained | 20 | 20% | 12.5 | 8% |
| Gemma 4 31B (teacher) | 3 | 2/3 | 18.7 | 0.2% |
