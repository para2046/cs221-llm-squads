"""HTTP server that runs N SMACv2 games (one SC2 process each) and speaks only text.

The GPU learner never touches StarCraft: it POSTs /reset and /step and gets back, for every
living agent, the text prompt; it sends back one reply string per agent. Run it inside the CPU
Slurm job (SC2 needs CPUs, not GPUs):

  srun --jobid=<cpu job> --overlap --cpus-per-task=16 env SC2PATH=... \
      .venv-smac/bin/python -m smac_llm.env_server --n-envs 8 --port 8390 --token-file <file>
"""
from __future__ import annotations

import argparse
import json
import os
import multiprocessing as mp
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _worker(conn, env_args: dict, seed: int, game_args: dict):
    from . import game as game_mod
    from .game import Game

    game_args = dict(game_args)
    game_mod.STATE_OUT[0] = bool(game_args.pop("global_state", False))
    game = Game(env_args, seed, **game_args)
    replay_dir, scenario = env_args.get("replay_dir"), None
    while True:
        cmd, data = conn.recv()
        try:
            if cmd == "reset":
                scenario = (data or {}).get("scenario")
                conn.send(game.reset(scenario))
            elif cmd == "step":
                out = game.step({int(k): v for k, v in data["replies"].items()},
                                {int(k): v for k, v in (data.get("replies_b") or {}).items()} or None)
                if replay_dir and out.get("done"):
                    _save_replay(game, replay_dir, scenario, out)
                conn.send(out)
            elif cmd == "close":
                game.close()
                conn.send({"ok": True})
                return
        except Exception as e:  # report, keep the worker alive
            import traceback
            conn.send({"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]})


def _save_replay(game, replay_dir: str, scenario, out: dict):
    """Save the finished game's SC2 replay as scen<S>_<winner>_s<steps>_<time>.SC2Replay (only with --replay-dir)."""
    import glob, os, time
    try:
        before = set(glob.glob(os.path.join(replay_dir, "*.SC2Replay")))
        game.env.save_replay()
        info = out.get("info") or {}
        res = "won" if info.get("battle_won") else (info.get("winner") or "lost")
        for f in set(glob.glob(os.path.join(replay_dir, "*.SC2Replay"))) - before:
            os.rename(f, os.path.join(replay_dir, f"scen{scenario}_{res}_s{out.get('step')}_{time.strftime('%H%M%S')}.SC2Replay"))
    except Exception as e:  # never break the game over a replay
        print(f"replay save failed: {e}", flush=True)


class EnvPool:
    def __init__(self, n_envs: int, env_args: dict, base_seed: int, game_args: dict):
        ctx = mp.get_context("spawn")
        self.conns, self.locks, self.procs = [], [], []
        for i in range(n_envs):
            parent, child = ctx.Pipe()
            ea = env_args if not env_args.get("replay_dir") else {**env_args, "replay_dir": f"{env_args['replay_dir']}/env{i}"}
            p = ctx.Process(target=_worker, args=(child, ea, base_seed + 101 * i, game_args), daemon=True)
            p.start()
            self.conns.append(parent)
            self.locks.append(threading.Lock())
            self.procs.append(p)

    def call(self, i: int, cmd: str, data=None):
        with self.locks[i]:
            self.conns[i].send((cmd, data))
            return self.conns[i].recv()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--port", type=int, default=8390)
    p.add_argument("--token-file", required=True)
    p.add_argument("--map", default="10gen_protoss")
    p.add_argument("--n-units", type=int, default=5)
    p.add_argument("--n-enemies", type=int, default=None)
    p.add_argument("--difficulty", default="7")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--opponent", default="builtin",
                   help="builtin | external (self-play: the trainer also plays team B) | random | scripted | "
                        "kite_local | fixed:<action> | llm:<base_url>:<model>")
    p.add_argument("--opponent-key-env", default=None, help="env var with the opponent's API key")
    p.add_argument("--channel", choices=["none", "team", "all", "struct", "info", "vision"], default="none")
    p.add_argument("--no-history", action="store_true", help="unified protocol without the per-unit 5-step history")
    p.add_argument("--brief", choices=["none", "rules", "facts", "tactics", "unified"], default="none")
    p.add_argument("--incentive-a", default=None, help="extra system-prompt text for the learner's team")
    p.add_argument("--incentive-b", default=None, help="extra system-prompt text for the opponent team")
    p.add_argument("--msg-ablation", choices=["none", "drop", "shuffle"], default="none")
    p.add_argument("--replay-dir", default=None, help="save each finished game's SC2 replay under DIR/env<i>/")
    p.add_argument("--reward", choices=["smac", "shaped", "shaped2", "shaped21"], default="smac",
                   help="training reward for team A (stats always report the SMAC reward too)")
    p.add_argument("--global-state", action="store_true", help="return the SMAC global state (centralised critic)")
    args = p.parse_args()
    token = open(args.token_file).read().strip()
    env_args = dict(map=args.map, n_units=args.n_units, n_enemies=args.n_enemies,
                    difficulty=args.difficulty,
                    replay_dir=os.path.abspath(args.replay_dir) if args.replay_dir else None)
    game_args = dict(opponent=args.opponent, channel=args.channel, incentive_a=args.incentive_a,
                     incentive_b=args.incentive_b, opponent_key_env=args.opponent_key_env,
                     msg_ablation=args.msg_ablation, reward_mode=args.reward, brief=args.brief,
                     history=not args.no_history)
    if args.global_state:
        game_args["global_state"] = True
    pool = EnvPool(args.n_envs, env_args, args.seed, game_args)
    from .game import system_prompt
    info = {"n_envs": args.n_envs, **env_args, "opponent": args.opponent, "channel": args.channel,
            "msg_ablation": args.msg_ablation, "reward": args.reward,
            "brief": args.brief, "system": system_prompt(args.channel, args.incentive_a, args.brief, builtin_enemy=args.opponent == "builtin",
                                   history=not args.no_history), "history": not args.no_history}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _reply(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.headers.get("Authorization") != f"Bearer {token}":
                return self._reply(401, {"error": "unauthorized"})
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path == "/info":
                return self._reply(200, info)
            if self.path in ("/reset", "/step"):
                return self._reply(200, pool.call(int(data["env"]), self.path[1:], data))  # reset: optional "scenario"
            self._reply(404, {"error": "unknown path"})

    class Server(ThreadingHTTPServer):
        request_queue_size = 256  # default 5 drops connections when all envs step at once
        daemon_threads = True

    srv = Server(("0.0.0.0", args.port), H)
    print(f"env server ready host={socket.gethostname()} port={args.port} n_envs={args.n_envs}",
          flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
