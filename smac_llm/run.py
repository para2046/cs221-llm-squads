"""Play SMAC episodes with random or LLM agents and log every decision as JSONL.

Each agent is a separate call (decentralised execution, same weights = cooperative self-play).
The game is lock-step: SC2 waits until all actions for a step are in, so model latency only
costs wall-clock time, never game time.

Examples
  python -m smac_llm.run --map 3m --policy random --episodes 2
  python -m smac_llm.run --map 3m --policy llm --base-url http://127.0.0.1:8000/v1 \
      --model gemma4b --episodes 5 --out runs/smoke_gemma4b.jsonl
  # frontier teacher through OpenRouter (key from $OPENROUTER_API_KEY)
  python -m smac_llm.run --map 3m --policy llm --base-url https://openrouter.ai/api/v1 \
      --model qwen/qwen3-235b-a22b-2507 --api-key-env OPENROUTER_API_KEY --episodes 5
"""
from __future__ import annotations

import argparse
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from smacv2.env import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

UNITS = {"terran": (["marine", "marauder", "medivac"], ["medivac"]),
         "protoss": (["stalker", "zealot", "colossus"], ["colossus"]),
         "zerg": (["zergling", "hydralisk", "baneling"], ["baneling"])}


def make_env(args):
    """SMACv2 native maps (10gen_<race>): random unit mix + spawn layout every episode, and a
    0.1 s reset. Legacy v1 maps (3m, 2s3z, ...) still work but reset takes minutes under SMACv2."""
    kw = dict(difficulty=args.difficulty, seed=args.seed)
    if args.replay_dir:
        kw.update(replay_dir=os.path.abspath(args.replay_dir), replay_prefix=args.map)
    if not args.map.startswith("10gen_"):
        env = StarCraft2Env(map_name=args.map, **kw)
        return env, env
    race = args.map.split("_")[1]
    unit_types, exceptions = UNITS[race]
    cfg = {"n_units": args.n_units, "n_enemies": args.n_enemies or args.n_units,
           "team_gen": {"dist_type": "weighted_teams", "unit_types": unit_types,
                        "weights": [0.45, 0.45, 0.1], "exception_unit_types": exceptions,
                        "observe": True},
           "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5,
                               "n_enemies": args.n_enemies or args.n_units,
                               "map_x": 32, "map_y": 32}}
    wrapper = StarCraftCapabilityEnvWrapper(
        capability_config=cfg, map_name=args.map, conic_fov=False, use_unit_ranges=True,
        min_attack_range=2, obs_own_pos=True, fully_observable=False, **kw)
    return wrapper, wrapper.env

from .text_env import COMM_SUFFIX, STRATEGY_PROMPT, SYSTEM_PROMPT, TextSMAC, snapshot

EXPLAIN_SUFFIX = (" Also add a short \"note\" field (at most 12 words) stating your intent, e.g. "
                  "{\"action\": \"attack_e1\", \"note\": \"focus the weakest stalker with a1\"}.")


class RandomPolicy:
    def act(self, tenv, agent_ids, prompts):
        out = {}
        for a in agent_ids:
            menu = tenv.action_menu(a)
            name = random.choice(list(menu))
            out[a] = (json.dumps({"action": name}), None)
        return out


_RPM = float(os.environ.get("LLM_RPM", "0"))  # per-process request budget per minute (0 = unlimited)
_rate_lock = threading.Lock()
_next_slot = [0.0]


def _throttle():
    """Space requests evenly so this process stays under LLM_RPM requests per minute."""
    if _RPM <= 0:
        return
    with _rate_lock:
        t = max(time.time(), _next_slot[0])
        _next_slot[0] = t + 60.0 / _RPM
    time.sleep(max(0.0, t - time.time()))


class LLMPolicy:
    def __init__(self, base_url, model, api_key, temperature=0.7, max_tokens=64, workers=16,
                 explain=False, reasoning_effort=None, comm=False, strategy=False):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY", timeout=20, max_retries=0)
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.system = SYSTEM_PROMPT + (EXPLAIN_SUFFIX if explain else "") + (COMM_SUFFIX if comm else "") \
            + (STRATEGY_PROMPT if strategy else "")
        if not reasoning_effort:
            self.extra = None
        elif "openrouter" in base_url:
            self.extra = {"reasoning": {"effort": reasoning_effort}}
        else:  # OpenAI-style servers (e.g. the CS221 proxy)
            self.extra = {"reasoning_effort": reasoning_effort}
        self.pool = ThreadPoolExecutor(workers)
        self.tokens = [0, 0]  # prompt, completion

    def _one(self, prompt):
        t0 = time.time()
        for attempt in range(40):  # retry rate limits / transient server errors with backoff
            _throttle()
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": self.system},
                              {"role": "user", "content": prompt}],
                    temperature=self.temperature, max_tokens=self.max_tokens,
                    extra_body=self.extra)
                text = r.choices[0].message.content or ""
                u = getattr(r, "usage", None)
                if u:
                    self.tokens[0] += u.prompt_tokens or 0
                    self.tokens[1] += u.completion_tokens or 0
                break
            except Exception as e:  # network/model errors become invalid actions, not crashes
                text = f"ERROR: {type(e).__name__}: {e}"
                if "429" in str(e):  # rate limited: wait for the window to reset, never count it as a move
                    time.sleep(5)
                    continue
                if attempt >= 4 or not any(c in str(e) for c in ("500", "502", "503", "504", "timeout", "Timeout",
                                                                 "timed out", "Connection")):
                    break
                time.sleep(1 + attempt)
        return text, round(time.time() - t0, 3)

    def act(self, tenv, agent_ids, prompts):
        futs = {a: self.pool.submit(self._one, prompts[a]) for a in agent_ids}
        return {a: f.result() for a, f in futs.items()}


def play_episode(runner, env, tenv, policy, ep, log):
    runner.reset()
    terminated, step, ep_reward, n_calls, n_invalid = False, 0, 0.0, 0, 0
    info = {}
    while not terminated:
        alive = [a for a in range(env.n_agents) if env.get_unit_by_id(a).health > 0]
        prompts = {a: tenv.describe(a, step) for a in alive}
        units_before = snapshot(env, tenv)
        replies = policy.act(tenv, alive, prompts)
        actions, records = [], []
        for a in range(env.n_agents):
            if a not in alive:
                actions.append(0)  # no-op for dead units
                continue
            text, latency = replies[a]
            act, valid = tenv.parse(a, text)
            actions.append(act)
            n_calls += 1
            n_invalid += (not valid)
            records.append({"agent": a, "prompt": prompts[a], "reply": text,
                            "action": act, "valid": valid, "latency": latency})
        reward, terminated, info = runner.step(actions)
        ep_reward += reward
        log.write(json.dumps({"episode": ep, "step": step, "reward": reward,
                              "units": units_before, "agents": records}) + "\n")
        step += 1
    summary = {"episode": ep, "summary": True, "won": bool(info.get("battle_won", False)),
               "return": round(ep_reward, 3), "steps": step,
               "dead_allies": info.get("dead_allies"), "dead_enemies": info.get("dead_enemies"),
               "invalid_rate": round(n_invalid / max(n_calls, 1), 3)}
    log.write(json.dumps(summary) + "\n")
    log.flush()
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="10gen_protoss")
    p.add_argument("--n-units", type=int, default=5)
    p.add_argument("--n-enemies", type=int, default=None)
    p.add_argument("--difficulty", default="7", help="built-in AI, '1' (very easy) .. '7' (very hard)")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--policy", choices=["random", "llm", "scripted", "kite", "focus_local", "kite_local"], default="random",
                   help="scripted: focus weakest in range else approach; kite: + step back during cooldown")
    p.add_argument("--strategy", action="store_true", help="append the rules+tactics brief to the system prompt")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="gemma4b")
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--explain", action="store_true", help="ask for a short intent note (demo only)")
    p.add_argument("--reasoning-effort", default=None, help="OpenRouter reasoning effort, e.g. low")
    p.add_argument("--out", default="runs/smac_smoke.jsonl")
    p.add_argument("--replay-dir", default=None, help="save SC2 replays here")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    runner, env = make_env(args)
    tenv = TextSMAC(env)
    if args.policy in ("scripted", "kite", "focus_local", "kite_local"):
        from .game import KitePolicy, ScriptedPolicy
        policy = {"scripted": ScriptedPolicy, "kite": KitePolicy,
                  "focus_local": lambda: KitePolicy(local=True, kite=False),
                  "kite_local": lambda: KitePolicy(local=True)}[args.policy]()
    else:
        policy = RandomPolicy() if args.policy == "random" else LLMPolicy(
            args.base_url, args.model,
            os.environ.get(args.api_key_env) if args.api_key_env else None, args.temperature,
            args.max_tokens, explain=args.explain, reasoning_effort=args.reasoning_effort,
            strategy=args.strategy)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results = []
    with open(args.out, "a") as log:
        for ep in range(args.episodes):
            t0 = time.time()
            s = play_episode(runner, env, tenv, policy, ep, log)
            s["seconds"] = round(time.time() - t0, 1)
            print(json.dumps(s), flush=True)
            results.append(s)
            if args.replay_dir:
                env.save_replay()
    runner.close()
    n = len(results)
    print(json.dumps({"map": args.map, "n_units": args.n_units, "difficulty": args.difficulty, "policy": args.policy,
                      "model": args.model if args.policy == "llm" else None, "episodes": n,
                      "win_rate": sum(r["won"] for r in results) / n,
                      "mean_return": round(sum(r["return"] for r in results) / n, 3),
                      "mean_invalid_rate": round(sum(r["invalid_rate"] for r in results) / n, 3)}))


if __name__ == "__main__":
    main()
