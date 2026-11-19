"""Play logged games with smac_llm.game.Game: team A is any policy, team B is the built-in AI or any
policy (self-play = the same spec on both sides). All prompt/comm options of Game are available.

usage:
  python -m smac_llm.play --a llm:https://.../v1:openai/gpt-oss-120b --api-key-env CS221_API_KEY \
      --opponent builtin --brief facts --channel struct --episodes 4 --out runs/X/w0.jsonl
  python -m smac_llm.play --a <spec> --opponent <same spec> ...      # self-play
Log lines: {"episode","step","units","team_a":[...],"team_b":[...]} and one summary per episode.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

from .game import Game, make_opponent
from .text_env import snapshot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="team A policy spec (see game.make_opponent)")
    p.add_argument("--opponent", default="builtin", help="builtin | policy spec for team B")
    p.add_argument("--api-key-env", default=None, help="env var with the API key (both teams)")
    p.add_argument("--brief", choices=["none", "rules", "facts", "tactics", "unified"], default="none")
    p.add_argument("--channel", choices=["none", "team", "struct", "info", "vision"], default="none")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--reasoning-effort", default=None, help="e.g. low (for reasoning models)")
    p.add_argument("--map", default="10gen_protoss")
    p.add_argument("--n-units", type=int, default=5)
    p.add_argument("--difficulty", default="7")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--replay-dir", default=None)
    a = p.parse_args()

    random.seed(a.seed)
    env_args = dict(map=a.map, n_units=a.n_units, n_enemies=None, difficulty=a.difficulty, replay_dir=a.replay_dir)
    g = Game(env_args, a.seed, opponent=a.opponent, channel=a.channel, brief=a.brief, opponent_key_env=a.api_key_env,
             opponent_kwargs={"temperature": a.temperature, "max_tokens": a.max_tokens,
                                     "reasoning_effort": a.reasoning_effort})
    pol = make_opponent(a.a, a.channel, None, a.api_key_env, a.temperature, a.max_tokens, brief=a.brief,
                        reasoning_effort=a.reasoning_effort)
    if hasattr(pol, "system"):
        pol.system = g.system_a
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "a") as log:
        for ep in range(a.episodes):
            t0 = time.time()
            st = g.reset()
            n = bad = n_b = bad_b = 0
            ret = 0.0
            while not st["done"]:
                units = snapshot(g.env, g.tx_a)
                prompts = {int(k): v for k, v in st["prompts"].items()}
                rep = pol.act(g.tx_a, list(prompts), prompts)
                st = g.step({k: rep[k][0] for k in prompts})
                ret += st["reward"]
                recs = []
                for k in prompts:
                    ok = st["valid"].get(str(k), True)
                    n += 1; bad += not ok
                    r = {"agent": k, "prompt": prompts[k], "reply": rep[k][0], "latency": rep[k][1],
                         "action": st["actions_a"][str(k)], "valid": ok}
                    if a.channel == "struct":
                        r.update((st["info"].get("intents") or {}).get("A", {}).get(str(k), {}))
                    recs.append(r)
                tb = st.get("team_b") or []
                n_b += len(tb); bad_b += sum(not r["valid"] for r in tb)
                log.write(json.dumps({"episode": ep, "step": st["step"] - 1, "reward": st["reward"], "units": units,
                                      "team_a": recs, "team_b": tb}) + "\n")
                log.flush()
            info = st["info"]
            won = bool(info.get("battle_won"))
            winner = info.get("winner") or ("A" if won else ("B" if info.get("dead_allies") == a.n_units else "draw"))
            s = {"episode": ep, "summary": True, "won": won, "winner": winner, "return": round(ret, 3),
                 "return_b": info.get("return_b"), "steps": st["step"], "dead_allies": info.get("dead_allies"),
                 "dead_enemies": info.get("dead_enemies"), "invalid_rate": round(bad / max(n, 1), 3),
                 "invalid_rate_b": round(bad_b / max(n_b, 1), 3) if n_b else None,
                 "tokens_a": getattr(pol, "tokens", None), "seconds": round(time.time() - t0, 1)}
            log.write(json.dumps(s) + "\n"); log.flush()
            print(json.dumps(s), flush=True)
            if a.replay_dir:
                g.env.save_replay()
    g.close()


if __name__ == "__main__":
    main()
