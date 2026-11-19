"""Model vs model on SMACv2: team A (SC2 player 1) and team B (player 2) are each run by an LLM
(or random policy); the built-in AI is removed.

SMACv2 always creates "agent vs. built-in computer". Here we start two SC2 processes, join both
to one two-player game, and keep them in lock-step. Everything else in SMACv2 is reused: the
units are still spawned with debug commands (owner 1 / owner 2), and SMACv2's bookkeeping is
from team A's view. Team B sees the same text interface with allies and enemies swapped.

Optional message channel (--channel): each unit may add {"say": "..."} to its reply; the message
is shown to every living unit of its own team (team) or of both teams (all) on the next step.
Units never see each other's prompts or replies otherwise.

Example
  python -m smac_llm.versus --a random --b random --episodes 2 --out runs/versus_smoke.jsonl
  python -m smac_llm.versus --a llm:https://openrouter.ai/api/v1:google/gemma-4-31b-it \
      --b llm:https://openrouter.ai/api/v1:qwen/qwen3-235b-a22b-2507 --api-key-env OPENROUTER_API_KEY \
      --channel team --episodes 10 --out runs/versus_gemma31b_qwen.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pysc2 import maps as _real_maps
from pysc2 import run_configs as _real_run_configs
from pysc2.lib import portspicker
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import raw_pb2 as r_pb
from s2clientprotocol import sc2api_pb2 as sc_pb
from smacv2.env.starcraft2 import starcraft2 as sc2mod

from .run import LLMPolicy, RandomPolicy, make_env
from .text_env import TextSMAC, parse_say, snapshot

# (shoot range, sight range) by unit name, as in SMACv2's use_unit_ranges tables
RANGES = {"stalker": (6, 10), "zealot": (0.1, 9), "colossus": (7, 10), "zergling": (0.1, 8),
          "baneling": (0.25, 8), "hydralisk": (5, 9), "marine": (5, 9), "marauder": (6, 10),
          "medivac": (4, 11)}


# ---------------------------------------------------------------------------- two-player SC2
class DualController:
    """Looks like SMACv2's single controller; drives both players' controllers in lock-step."""

    def __init__(self, env, c1, c2, ports):
        self.env, self.c1, self.c2, self.ports = env, c1, c2, ports
        self.pool = ThreadPoolExecutor(2)
        self.pending_b: list = []  # player-2 actions to send with the next actions() call

    def _both(self, f1, f2):
        fut = self.pool.submit(f2)
        r = f1()
        fut.result()
        return r

    def create_game(self, req):
        del req.player_setup[:]
        req.player_setup.add(type=sc_pb.Participant)
        req.player_setup.add(type=sc_pb.Participant)
        # multiplayer: both games need the map on disk; pass only its path (as pysc2 does)
        path, data = req.local_map.map_path, req.local_map.map_data
        for c in (self.c1, self.c2):
            c.save_map(path, data)
        req.local_map.ClearField("map_data")
        return self.c1.create_game(req)

    def join_game(self, join1):
        join2 = sc_pb.RequestJoinGame()
        join2.CopyFrom(join1)
        join2.race = sc2mod.races[self.env._bot_race]
        for j in (join1, join2):
            j.shared_port = 0
            j.server_ports.game_port, j.server_ports.base_port = self.ports[0], self.ports[1]
            del j.client_ports[:]
            j.client_ports.add(game_port=self.ports[2], base_port=self.ports[3])
        return self._both(lambda: self.c1.join_game(join1), lambda: self.c2.join_game(join2))

    def actions(self, req):
        b = sc_pb.RequestAction(actions=self.pending_b)
        self.pending_b = []
        return self._both(lambda: self.c1.actions(req), lambda: self.c2.actions(b))

    def step(self, n=1):
        return self._both(lambda: self.c1.step(n), lambda: self.c2.step(n))

    def __getattr__(self, name):  # observe, debug, game_info, data, save_replay, ...
        return getattr(self.c1, name)


class _Procs:
    def __init__(self, p1, p2, controller):
        self.p1, self.p2, self.controller = p1, p2, controller

    def close(self):
        for p in (self.p1, self.p2):
            try:
                p.close()
            except Exception:
                pass
        portspicker.return_ports(self.controller.ports)


class _RunConfigShim:
    """Stands in for pysc2's run config inside SMACv2's _launch: start() launches two games."""

    def __init__(self, rc, env):
        self.rc, self.env = rc, env

    def __getattr__(self, name):
        return getattr(self.rc, name)

    def start(self, **kw):
        ports = portspicker.pick_unused_ports(4)
        p1 = self.rc.start(extra_ports=ports, **kw)
        p2 = self.rc.start(extra_ports=ports, **kw)
        return _Procs(p1, p2, DualController(self.env, p1.controller, p2.controller, ports))


class _RunConfigsShim:
    def __init__(self, env):
        self.env = env

    def get(self, **kw):
        return _RunConfigShim(_real_run_configs.get(**kw), self.env)


class _MapsShim:
    """SMAC maps declare player 2 as a computer slot, so a second client cannot join. We use a
    copy whose MapInfo marks player 2 as a user slot (see scripts/make_2p_map.py)."""

    def get(self, name):
        m = _real_maps.get(name)

        class _M:
            path = m.path.replace(".SC2Map", "_2p.SC2Map")
        return _M()


def make_versus(args):
    """SMACv2 env (same capability config as run.py) whose _launch starts a two-player game."""
    runner, env = make_env(args)
    orig_launch = env._launch

    def launch():
        real_rc, real_maps = sc2mod.run_configs, sc2mod.maps
        sc2mod.run_configs, sc2mod.maps = _RunConfigsShim(env), _MapsShim()
        try:
            orig_launch()
        finally:
            sc2mod.run_configs, sc2mod.maps = real_rc, real_maps

    env._launch = launch
    # Spawn team B with the same "_RL" unit types as team A. SMAC gives the enemy stock units,
    # which auto-acquire targets; the _RL units are passive (they only act when ordered), so
    # with stock units on one side the two teams would not play the same game.
    env._convert_unit_name_to_unit_type = lambda unit_name, ally=True: env.ally_unit_map[unit_name]
    return runner, env


# ---------------------------------------------------------------------------- team B view
class TeamBView:
    """The env seen from player 2: allies <-> enemies, so TextSMAC works unchanged."""

    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def agents(self):
        return self.env.enemies

    @property
    def enemies(self):
        return self.env.agents

    @property
    def n_agents(self):
        return self.env.n_enemies

    def get_unit_by_id(self, i):
        return self.env.enemies[i]

    def _ranges(self, i):
        shoot, sight = RANGES.get(TextSMAC(self.env)._own_type_name(self.env.enemies[i].unit_type), (6, 9))
        return max(shoot, self.env.min_attack_range), sight

    def unit_shoot_range(self, i):
        return self._ranges(i)[0]

    def unit_sight_range(self, i):
        return self._ranges(i)[1]

    def get_avail_agent_actions(self, i):
        env, unit = self.env, self.env.enemies[i]
        avail = [0] * env.n_actions
        if unit.health <= 0:
            avail[0] = 1
            return avail
        avail[1] = 1
        for a, d in ((2, sc2mod.Direction.NORTH), (3, sc2mod.Direction.SOUTH),
                     (4, sc2mod.Direction.EAST), (5, sc2mod.Direction.WEST)):
            avail[a] = int(env.can_move(unit, d))
        rng = self.unit_shoot_range(i)
        for t_id, t in env.agents.items():
            if t.health > 0 and math.dist((unit.pos.x, unit.pos.y), (t.pos.x, t.pos.y)) <= rng:
                avail[t_id + env.n_actions_no_attack] = 1
        return avail

    def raw_action(self, i, action):
        """SC2 raw command for team-B unit i (mirrors StarCraft2Env.get_agent_action)."""
        env, unit = self.env, self.env.enemies[i]
        if action == 0 or unit.health <= 0:
            return None
        x, y, m = unit.pos.x, unit.pos.y, env._move_amount
        if action == 1:
            cmd = r_pb.ActionRawUnitCommand(ability_id=sc2mod.actions["stop"], unit_tags=[unit.tag])
        elif 2 <= action <= 5:
            tx, ty = {2: (x, y + m), 3: (x, y - m), 4: (x + m, y), 5: (x - m, y)}[action]
            cmd = r_pb.ActionRawUnitCommand(ability_id=sc2mod.actions["move"], unit_tags=[unit.tag],
                                            target_world_space_pos=sc_common.Point2D(x=tx, y=ty))
        else:
            target = env.agents[action - env.n_actions_no_attack]
            cmd = r_pb.ActionRawUnitCommand(ability_id=sc2mod.actions["attack"], unit_tags=[unit.tag],
                                            target_unit_tag=target.tag)
        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))


def team_damage(prev: dict, cur: dict, death_value: float) -> float:
    """Damage dealt to the units in `cur` since `prev` (+ death_value per kill), SMAC-style."""
    r = 0.0
    for k, p in prev.items():
        if p.health <= 0:
            continue
        c = cur[k]
        if c.health <= 0:
            r += p.health + p.shield + death_value
        else:
            r += max(p.health + p.shield - c.health - c.shield, 0)
    return r


# ---------------------------------------------------------------------------- game loop
class FixedPolicy:
    """Diagnostic policy: always give the same order (e.g. 'stop') to every unit."""

    def __init__(self, name):
        self.name = name

    def act(self, tenv, agent_ids, prompts):
        return {a: (json.dumps({"action": self.name}), None) for a in agent_ids}


def make_policy(spec: str, args):
    if spec == "random":
        return RandomPolicy()
    if spec.startswith("fixed:"):
        return FixedPolicy(spec.split(":", 1)[1])
    kind, rest = spec.split(":", 1)
    assert kind == "llm", f"policy spec must be 'random' or 'llm:<base_url>:<model>', got {spec}"
    base_url, model = rest.rsplit(":", 1)
    key = os.environ.get(args.api_key_env) if args.api_key_env else None
    return LLMPolicy(base_url, model, key, args.temperature, args.max_tokens,
                     comm=args.channel != "none", reasoning_effort=args.reasoning_effort)


def play_versus(runner, env, pol_a, pol_b, ep, log, channel="none"):
    runner.reset()
    tb = TeamBView(env)
    teams = {"A": (TextSMAC(env), env, pol_a), "B": (TextSMAC(tb), tb, pol_b)}
    inbox = {"A": [], "B": []}  # messages produced last step, visible this step
    ret = {"A": 0.0, "B": 0.0}
    calls = {"A": 0, "B": 0}
    invalid = {"A": 0, "B": 0}
    terminated, step, info = False, 0, {}
    max_reward = env.max_reward / env.reward_scale_rate if env.reward_scale else 1.0
    while not terminated:
        alive = {t: [a for a in view.agents if view.agents[a].health > 0]
                 for t, (_, view, _) in teams.items()}
        prompts, futures = {}, {}
        for t, (tx, view, pol) in teams.items():
            msgs = None if channel == "none" else inbox[t]
            prompts[t] = {a: tx.describe(a, step, msgs) for a in alive[t]}
        units_before = snapshot(env, teams["A"][0])
        with ThreadPoolExecutor(2) as ex:  # both teams think at the same time
            for t, (tx, view, pol) in teams.items():
                futures[t] = ex.submit(pol.act, tx, alive[t], prompts[t])
            replies = {t: f.result() for t, f in futures.items()}

        records, acts, said = {"A": [], "B": []}, {}, {"A": [], "B": []}
        for t, (tx, view, pol) in teams.items():
            acts[t] = []
            for a in range(view.n_agents):
                if a not in alive[t]:
                    acts[t].append(0)
                    continue
                text, latency = replies[t][a]
                act, valid = tx.parse(a, text)
                acts[t].append(act)
                calls[t] += 1
                invalid[t] += not valid
                say = parse_say(text) if channel != "none" else None
                if say:
                    said[t].append(f"{t.lower()}{a}: {say}" if channel == "all" else f"a{a}: {say}")
                records[t].append({"agent": a, "prompt": prompts[t][a], "reply": text,
                                   "action": act, "valid": valid, "latency": latency, "say": say})
        if channel == "team":
            inbox = said
        elif channel == "all":
            inbox = {"A": said["A"] + said["B"], "B": said["A"] + said["B"]}

        prev_a = {k: _copy_hp(u) for k, u in env.agents.items()}
        env._controller.pending_b = [c for a, act in enumerate(acts["B"])
                                     if (c := tb.raw_action(a, act)) is not None]
        reward_a, terminated, info = runner.step(acts["A"])
        reward_b = team_damage(prev_a, env.agents, env.reward_death_value)
        a_alive = sum(u.health > 0 for u in env.agents.values())
        b_alive = sum(u.health > 0 for u in env.enemies.values())
        b_won = terminated and a_alive == 0 and b_alive > 0
        if b_won:
            reward_b += env.reward_win
        ret["A"] += reward_a
        ret["B"] += reward_b / max_reward
        log.write(json.dumps({"episode": ep, "step": step, "reward_a": reward_a,
                              "reward_b": round(reward_b / max_reward, 4), "units": units_before,
                              "team_a": records["A"], "team_b": records["B"]}) + "\n")
        step += 1
    a_won = bool(info.get("battle_won", False))
    winner = "A" if a_won else ("B" if b_won else "draw")
    summary = {"episode": ep, "summary": True, "winner": winner, "steps": step,
               "return_a": round(ret["A"], 3), "return_b": round(ret["B"], 3),
               "alive_a": a_alive, "alive_b": b_alive,
               "invalid_a": round(invalid["A"] / max(calls["A"], 1), 3),
               "invalid_b": round(invalid["B"] / max(calls["B"], 1), 3)}
    log.write(json.dumps(summary) + "\n")
    log.flush()
    return summary


class _HP:
    __slots__ = ("health", "shield")

    def __init__(self, h, s):
        self.health, self.shield = h, s


def _copy_hp(u):
    return _HP(u.health, getattr(u, "shield", 0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--map", default="10gen_protoss")
    p.add_argument("--n-units", type=int, default=5)
    p.add_argument("--n-enemies", type=int, default=None)
    p.add_argument("--a", default="random", help="'random' or 'llm:<base_url>:<model>'")
    p.add_argument("--b", default="random")
    p.add_argument("--swap", action="store_true", help="alternate which policy plays player 1")
    p.add_argument("--channel", choices=["none", "team", "all"], default="none")
    p.add_argument("--api-key-env", default=None)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--out", default="runs/versus_smoke.jsonl")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replay-dir", default=None, help="save an SC2 replay of every game here")
    args = p.parse_args()
    args.difficulty = "7"  # unused: there is no built-in AI

    random.seed(args.seed)
    runner, env = make_versus(args)
    pols = {"a": make_policy(args.a, args), "b": make_policy(args.b, args)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tally = {"a": 0, "b": 0, "draw": 0}
    with open(args.out, "a") as log:
        for ep in range(args.episodes):
            flip = args.swap and ep % 2 == 1
            p1, p2 = (pols["b"], pols["a"]) if flip else (pols["a"], pols["b"])
            t0 = time.time()
            s = play_versus(runner, env, p1, p2, ep, log, args.channel)
            # report from the policies' point of view, not the SC2 player slots
            s["a_is_player"] = 2 if flip else 1
            w = s["winner"]
            s["winner_policy"] = "draw" if w == "draw" else ("a" if (w == "A") != flip else "b")
            tally[s["winner_policy"]] += 1
            s["seconds"] = round(time.time() - t0, 1)
            if args.replay_dir:
                s["replay"] = env.save_replay()
            print(json.dumps(s), flush=True)
    runner.close()
    print(json.dumps({"a": args.a, "b": args.b, "channel": args.channel, "episodes": args.episodes,
                      "wins_a": tally["a"], "wins_b": tally["b"], "draws": tally["draw"]}))


if __name__ == "__main__":
    main()
