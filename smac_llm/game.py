"""One SMACv2 game as seen by a learner that controls team A, with every optional feature:

  opponent   'builtin'  SMAC's built-in AI (single SC2 client; the original setting)
             'random' | 'fixed:<action>' | 'scripted' | 'llm:<base_url>:<model>'
                        team B is a policy we run (two-player game, see versus.py)
  channel    'none'     units only see their own surroundings
             'team'     each unit may add {"say": ...}; teammates read it next step
             'all'      both teams read every message (setting for the collusion probe)
  incentive  extra system-prompt text for team A / team B (e.g. a secret side objective)
  scenario   reset(scenario=k) replays the same unit types and start positions for every
             call with the same k, so GRPO can compare several rollouts of one situation.

Used by env_server.py (training/eval over HTTP) and directly by scripts.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .text_env import BRIEFS, COMM_SUFFIX, STRUCT_SUFFIX, INFO_SUFFIX, SYSTEM_PROMPT, TextSMAC, parse_field, parse_say


SELFPLAY_ENEMY = ("ENEMY BEHAVIOUR\n- The enemy team is controlled by another AI player with the same units, "
                  "rules and information as you. It may focus fire, retreat or kite.\n")


def system_prompt(channel: str = "none", incentive: str | None = None, brief: str = "none",
                  builtin_enemy: bool = True, history: bool = True) -> str:
    """brief: 'none' | 'facts' (rules + unit data + enemy behaviour; tactics only as optional
    examples) | 'tactics' (rules + prescribed tactics). channel: 'none' | 'team'/'all' (free
    "say") | 'struct' (automatic JSON status reports + optional focus/plan) | 'vision' (reports only)."""
    if brief == "unified":  # same text for every opponent; channel blocks without strategy hints
        from .text_env import UNIFIED_CHANNEL
        assert channel in UNIFIED_CHANNEL, f"unified protocol supports channels {list(UNIFIED_CHANNEL)}"
        b = BRIEFS[brief] if history else BRIEFS[brief].split("\nYOUR HISTORY")[0]
        return SYSTEM_PROMPT + b + UNIFIED_CHANNEL[channel] + ("\n" + incentive.strip() if incentive else "")
    b = BRIEFS[brief]
    if not builtin_enemy and "ENEMY BEHAVIOUR" in b:  # model-vs-model: the scripted-enemy description is wrong
        head, rest = b.split("ENEMY BEHAVIOUR", 1)
        b = head + SELFPLAY_ENEMY + rest[rest.index("\n", rest.index("\n") + 1) + 1:]
    s = SYSTEM_PROMPT + b
    s += {"none": "", "team": COMM_SUFFIX, "all": COMM_SUFFIX, "struct": STRUCT_SUFFIX, "info": INFO_SUFFIX, "vision": INFO_SUFFIX}[channel]
    if channel == "all":
        s += " Messages are visible to BOTH teams."
    if incentive:
        s += "\n" + incentive.strip()
    return s


class ScriptedPolicy:
    """Hand-written baseline: focus the weakest enemy in range, else walk to the nearest one."""

    def act(self, tenv, agent_ids, prompts):
        out = {}
        for a in agent_ids:
            menu = tenv.action_menu(a)
            attacks = [(tenv.env.enemies[int(k[8:])].health + tenv.env.enemies[int(k[8:])].shield, k)
                       for k in menu if k.startswith("attack_e")]
            if attacks:
                name = min(attacks)[1]
            else:
                me = tenv.env.get_unit_by_id(a)
                alive = [u for u in tenv.env.enemies.values() if u.health > 0]
                if alive:
                    tgt = min(alive, key=lambda u: math.dist((me.pos.x, me.pos.y), (u.pos.x, u.pos.y)))
                    name = next((k for k, v in menu.items() if v == tenv.move_toward(a, tgt)), "stop")
                else:
                    name = "stop"
            out[a] = (json.dumps({"action": name}), None)
        return out


class KitePolicy:
    """Rule-based tactics from STRATEGY_PROMPT: team focus fire on the weakest enemy in range;
    while the weapon cools down, a unit that was just hit (or a stalker with a zealot close by)
    steps away from the enemy toward its teammates; otherwise approach as a group."""
    STEP = {"move_north": (0, 1), "move_south": (0, -1), "move_east": (1, 0), "move_west": (-1, 0)}

    def __init__(self, move_amount: float = 2.0, local: bool = False, kite: bool = True):
        """local=True: each unit only uses what its own observation shows (enemies and teammates
        within its sight range), like the LLM; no shared team target. With nothing in sight it
        walks toward the map centre. kite=False drops the step-back rule (plain focus fire)."""
        self.prev, self.move, self.local, self.kite = {}, move_amount, local, kite

    def act(self, tenv, agent_ids, prompts):
        from types import SimpleNamespace as NS
        env = tenv.env
        hp = lambda u: u.health + u.shield
        dist = lambda u, x, y: math.dist((u.pos.x, u.pos.y), (x, y))
        foes = {k: u for k, u in env.enemies.items() if u.health > 0}
        menus = {a: tenv.action_menu(a) for a in agent_ids}
        in_range = {int(k[8:]) for a in agent_ids for k in menus[a] if k.startswith("attack_e")}
        focus = min(in_range, key=lambda k: (hp(foes[k]), k)) if in_range else None
        me_all = [env.get_unit_by_id(a) for a in agent_ids]
        cx = sum(u.pos.x for u in me_all) / len(me_all)
        cy = sum(u.pos.y for u in me_all) / len(me_all)
        out = {}
        all_foes = foes
        for a in agent_ids:
            me, menu = env.get_unit_by_id(a), menus[a]
            hit = hp(me) < self.prev.get(a, hp(me)) - 1e-6
            self.prev[a] = hp(me)
            attacks = [int(k[8:]) for k in menu if k.startswith("attack_e")]
            if self.local:  # own observation only
                sight = env.unit_sight_range(a)
                foes = {k: u for k, u in all_foes.items() if dist(u, me.pos.x, me.pos.y) < sight}
                mates = [u for u in me_all if dist(u, me.pos.x, me.pos.y) < sight]
                cx = sum(u.pos.x for u in mates) / len(mates)
                cy = sum(u.pos.y for u in mates) / len(mates)
                focus = min(attacks, key=lambda k: (hp(all_foes[k]), k)) if attacks else None
            ready = me.weapon_cooldown <= 8  # fires within this step
            near = min(foes.values(), key=lambda u: dist(u, me.pos.x, me.pos.y)) if foes else None
            d_near = dist(near, me.pos.x, me.pos.y) if near else 99.0
            stalker = tenv._own_type_name(me.unit_type) == "stalker"
            zealot_close = near is not None and tenv._own_type_name(near.unit_type) == "zealot" and d_near < 3.5
            name = None
            if attacks and ready:
                tgt = focus if focus in attacks else min(attacks, key=lambda k: (hp(foes[k]), k))
                name = f"attack_e{tgt}"
            elif self.kite and near is not None and not ready and (hit or (stalker and zealot_close)) and d_near < 7:
                best = None
                for mv, (dx, dy) in self.STEP.items():
                    if mv not in menu:
                        continue
                    x, y = me.pos.x + dx * self.move, me.pos.y + dy * self.move
                    score = min(dist(u, x, y) for u in foes.values()) - 0.3 * math.dist((x, y), (cx, cy))
                    if best is None or score > best[0]:
                        best = (score, mv)
                name = best[1] if best else None
            if name is None and attacks:
                tgt = focus if focus in attacks else min(attacks, key=lambda k: (hp(foes[k]), k))
                name = f"attack_e{tgt}"
            if name is None and self.local and near is None:
                cx_map, cy_map = env.map_x / 2, env.map_y / 2
                act = tenv.move_toward(a, NS(pos=NS(x=cx_map, y=cy_map)))
                name = next((k for k, v in menu.items() if v == act), "stop")
            if name is None:
                if near is None or (math.dist((me.pos.x, me.pos.y), (cx, cy)) > 4 and d_near > 9):
                    goal = NS(pos=NS(x=cx, y=cy)) if near is None or math.dist((me.pos.x, me.pos.y), (cx, cy)) > 4 else near
                else:
                    goal = foes[focus] if focus is not None else near
                act = tenv.move_toward(a, goal)
                name = next((k for k, v in menu.items() if v == act), "stop")
            out[a] = (json.dumps({"action": name}), None)
        return out


def make_opponent(spec: str, channel: str, incentive: str | None, api_key_env: str | None,
                  temperature: float = 0.7, max_tokens: int = 64, brief: str = "none",
                  reasoning_effort: str | None = None):
    """Policy from a spec: random | scripted | kite | focus_local | kite_local | fixed:<action> |
    llm:<base_url>:<model>. Used for team B, and by play.py for team A."""
    from .run import LLMPolicy, RandomPolicy
    from .versus import FixedPolicy
    if spec == "random":
        return RandomPolicy()
    if spec == "scripted":
        return ScriptedPolicy()
    if spec in ("kite", "focus_local", "kite_local"):
        return {"kite": KitePolicy, "focus_local": lambda: KitePolicy(local=True, kite=False),
                "kite_local": lambda: KitePolicy(local=True)}[spec]()
    if spec.startswith("fixed:"):
        return FixedPolicy(spec.split(":", 1)[1])
    kind, rest = spec.split(":", 1)
    assert kind == "llm", spec
    base_url, model = rest.rsplit(":", 1)
    pol = LLMPolicy(base_url, model, os.environ.get(api_key_env) if api_key_env else None,
                    temperature, max(max_tokens, 96 if channel != "none" else max_tokens),
                    reasoning_effort=reasoning_effort)
    pol.system = system_prompt(channel, incentive, brief, builtin_enemy=False)
    return pol


def _hp(units):
    return {k: (u.health + u.shield) if u.health > 0 else 0.0 for k, u in units.items()}


class ShapedReward:
    """Training reward for team A (reward_mode='shaped'). Per step, for the whole team:
      hp     (enemy HP+shield lost)/enemy start total - (own HP+shield lost)/own start total
             (unlike SMAC's default, our own losses count; an episode sums to at most +-1)
      kill   +0.1 per enemy killed      death  -0.1 per own unit lost
      outcome at the end: win +1, loss -1, draw (time limit) -0.5
    Per unit (added to that unit's own decisions only):
      focus  +0.02 when it attacks the same target as a teammate, or its target dies this step
    Monte-Carlo returns would cancel a potential-based term (it telescopes), so the HP term is the
    plain per-step difference: it rewards ending ahead on HP, which is what we want."""
    W = dict(hp=1.0, kill=0.1, death=0.1, win=1.0, loss=1.0, draw=0.5, focus=0.005)

    def reset(self, env):
        self.a0 = sum(u.health_max + u.shield_max for u in env.agents.values()) or 1.0
        self.e0 = sum(u.health_max + u.shield_max for u in env.enemies.values()) or 1.0
        self.parts = dict.fromkeys(("hp", "kill", "death", "outcome", "focus"), 0.0)

    def __call__(self, env, prev_a, prev_e, acts, done, won, lost):
        W = self.W
        cur_a, cur_e = _hp(env.agents), _hp(env.enemies)
        dealt = sum(max(prev_e[k] - cur_e[k], 0.0) for k in cur_e) / self.e0
        taken = sum(max(prev_a[k] - cur_a[k], 0.0) for k in cur_a) / self.a0
        kills = sum(prev_e[k] > 0 and cur_e[k] <= 0 for k in cur_e)
        deaths = sum(prev_a[k] > 0 and cur_a[k] <= 0 for k in cur_a)
        parts = {"hp": W["hp"] * (dealt - taken), "kill": W["kill"] * kills, "death": -W["death"] * deaths,
                 "outcome": (W["win"] if won else -W["loss"] if lost else -W["draw"]) if done else 0.0}
        n_no_attack = env.n_actions_no_attack
        targets = {a: act - n_no_attack for a, act in enumerate(acts) if act >= n_no_attack and prev_a[a] > 0}
        own = {}
        for a, tg in targets.items():
            shared = sum(t == tg for t in targets.values()) >= 2
            finished = prev_e.get(tg, 0) > 0 and cur_e.get(tg, 0) <= 0
            if shared or finished:
                own[str(a)] = W["focus"]
        for k, v in parts.items():
            self.parts[k] += v
        self.parts["focus"] += sum(own.values())
        return sum(parts.values()), own


class ShapedReward2(ShapedReward):
    """Reward v2 (reward_mode='shaped2'; designed with Codex, 2026-11-17). Zero-sum between the teams.
    Team, per step:  D = 0.5 * dH + 0.25 * dL
      H = own (HP+shield)/own start total - enemy (HP+shield)/enemy start total   (signed: shield regen counts)
      L = (own alive - enemy alive) / n_units
    Outcome at the end: win +2, loss -2, draw 0. No focus / movement / talk bonus.
    Per unit (redistribution, sums to zero over the team): 0.5 * (c_u - D/n), where c_u splits D by who did what:
      each enemy's signed change (and kill) is split equally among the units that attacked it this step;
      each unit is charged its own signed change (and death); unattributed parts are spread over the units alive at the start of the step."""
    W2 = dict(hp=0.5, alive=0.25, win=2.0, loss=2.0, draw=0.0, credit=0.5)

    def reset(self, env):
        super().reset(env)
        self.n = max(len(env.agents), 1)
        self.parts = dict.fromkeys(("hp", "alive", "outcome", "credit_abs"), 0.0)

    def __call__(self, env, prev_a, prev_e, acts, done, won, lost):
        W, n = self.W2, self.n
        cur_a, cur_e = _hp(env.agents), _hp(env.enemies)
        n_no_attack = env.n_actions_no_attack
        attackers = {}
        for a, act in enumerate(acts):
            if act >= n_no_attack and prev_a.get(a, 0) > 0:
                attackers.setdefault(act - n_no_attack, []).append(a)
        c = dict.fromkeys(range(n), 0.0)
        unassigned = 0.0
        for k in cur_e:   # enemy losses are our gains
            g = W["hp"] * (prev_e[k] - cur_e[k]) / self.e0 + (W["alive"] / n if prev_e[k] > 0 and cur_e[k] <= 0 else 0.0)
            who = attackers.get(k, [])
            if who:
                for a in who:
                    c[a] += g / len(who)
            else:
                unassigned += g
        for k in cur_a:   # own losses (or shield regen) are charged to the unit itself
            c[k] = c.get(k, 0.0) + W["hp"] * (cur_a[k] - prev_a[k]) / self.a0 - (W["alive"] / n if prev_a[k] > 0 and cur_a[k] <= 0 else 0.0)
        live = [a for a in range(n) if prev_a.get(a, 0) > 0]
        for a in (live or list(c)):
            c[a] += unassigned / max(len(live), 1) if live else unassigned / n
        D = sum(c.values())
        outcome = (W["win"] if won else -W["loss"] if lost else -W["draw"]) if done else 0.0
        hp_part = W["hp"] * ((sum(cur_a.values()) - sum(prev_a.values())) / self.a0
                             - (sum(cur_e.values()) - sum(prev_e.values())) / self.e0)
        self.parts["hp"] += hp_part
        self.parts["alive"] += D - hp_part
        self.parts["outcome"] += outcome
        own = {}
        D_live = sum(c[a] for a in live)   # == D (non-live units have c = 0); explicit so sum(own) = 0 exactly
        for a in live:
            own[str(a)] = W["credit"] * (c[a] - D_live / len(live))
            self.parts["credit_abs"] += abs(own[str(a)])
        return D + outcome, own


class ShapedReward21(ShapedReward2):
    """Reward v2.1 (reward_mode='shaped21') = v2 + a one-off engagement-sync term (separate ablation, 2026-11-17).
    At first contact (first step any own and enemy unit are within R=9 of each other), with
    sync = share of a team's living units within R of some enemy, the team gets SYNC * (sync_own - sync_enemy).
    Same R for both teams, so the term is exactly zero-sum in self-play; bounded by +-SYNC (0.15)."""
    R, SYNC = 9.0, 0.15

    def reset(self, env):
        super().reset(env)
        self.contact = False
        self.parts["sync"] = 0.0

    @staticmethod
    def _sync(units, others, R):
        alive = [u for u in units.values() if u.health > 0]
        foes = [(u.pos.x, u.pos.y) for u in others.values() if u.health > 0]
        if not alive or not foes:
            return 0.0
        return sum(any(math.dist((u.pos.x, u.pos.y), f) <= R for f in foes) for u in alive) / len(alive)

    def __call__(self, env, prev_a, prev_e, acts, done, won, lost):
        team, own = super().__call__(env, prev_a, prev_e, acts, done, won, lost)
        if not self.contact:
            sa = self._sync(env.agents, env.enemies, self.R)
            if sa > 0:
                self.contact = True
                bonus = self.SYNC * (sa - self._sync(env.enemies, env.agents, self.R))
                self.parts["sync"] += bonus
                team += bonus
        return team, own


STATE_OUT = [False]  # env_server --global-state


class Game:
    def __init__(self, env_args: dict, seed: int, opponent: str = "builtin", channel: str = "none",
                 incentive_a: str | None = None, incentive_b: str | None = None,
                 opponent_key_env: str | None = None, msg_ablation: str = "none",
                 reward_mode: str = "smac", brief: str = "none", opponent_kwargs: dict | None = None,
                 history: bool = True):
        """msg_ablation (evaluation only): 'drop' hides every message from its readers while units
        are still told they can talk; 'shuffle' re-assigns each message to a random sender, so
        content survives but who-said-it does not. Comparing returns tests whether messages matter."""
        self.opponent, self.channel, self.msg_ablation = opponent, channel, msg_ablation
        self.shaper = {"shaped": ShapedReward, "shaped2": ShapedReward2, "shaped21": ShapedReward21}.get(reward_mode, lambda: None)()
        ns = argparse.Namespace(**{**env_args, "seed": seed})
        if opponent == "builtin":
            from .run import make_env
            self.runner, self.env = make_env(ns)
            self.tb = None
        else:
            from .versus import TeamBView, make_versus
            self.runner, self.env = make_versus(ns)
            self.tb = TeamBView(self.env)
            self.pol_b = None if opponent == "external" else make_opponent(
                opponent, channel, incentive_b, opponent_key_env, brief=brief, **(opponent_kwargs or {}))
            self.shaper_b = {"shaped": ShapedReward, "shaped2": ShapedReward2, "shaped21": ShapedReward21}.get(reward_mode, lambda: None)()
            self.tx_b = TextSMAC(self.tb)
            self.pool = ThreadPoolExecutor(1)
        self.tx_a = TextSMAC(self.env)
        self.brief = brief
        self.history = history and brief == "unified"
        self.system_a = system_prompt(channel, incentive_a, brief, builtin_enemy=opponent == "builtin", history=history)
        self.system_b = system_prompt(channel, incentive_b, brief, builtin_enemy=False)
        self.step_i = 0

    # ------------------------------------------------------------------ helpers
    def _seed_scenario(self, scenario: int):
        random.seed(scenario)
        dists = getattr(self.runner, "env_key_to_distribution_map", {})
        for k, d in enumerate(dists.values()):
            d.rng = np.random.default_rng(scenario * 1000 + k)

    def _alive(self, view):
        return [a for a in view.agents if view.agents[a].health > 0]

    # SC2 hides weapon cooldown of the other player's units, so team B's own cooldown reads 0.
    # Estimate it: after an attack order the weapon is busy for one period (game loops), then counts down 8/step.
    CD_PERIOD = {"zealot": 21, "stalker": 30, "colossus": 24}

    def _update_cd_b(self):
        if self.tb is None:
            return
        n0 = self.tb.n_actions_no_attack
        acts = getattr(self, "_last_acts_b", None) or []
        for b, u in self.tb.agents.items():
            cd = max(0, self.cd_b.get(b, 0) - 8)
            if b < len(acts) and acts[b] >= n0 and u.health > 0:
                cd = max(0, self.CD_PERIOD.get(self.tx_b._own_type_name(u.unit_type), 24) - 8) if self.cd_b.get(b, 0) <= 8 else cd
            self.cd_b[b] = cd
        self._last_acts_b = None

    def _apply_cd_b(self):
        for b, u in self.tb.agents.items():
            try:
                u.weapon_cooldown = float(self.cd_b.get(b, 0))
            except (AttributeError, TypeError):
                pass

    def _prompts(self, team):
        tx, view = (self.tx_a, self.env) if team == "A" else (self.tx_b, self.tb)
        if team == "B":
            self._apply_cd_b()
        alive = self._alive(view)
        if self.channel in ("struct", "info", "vision"):
            reps = {a: tx.team_report(a, self.intents[team].get(a) if self.channel == "struct" else None)
                    for a in alive}
            out = {a: tx.describe(a, self.step_i, reports=[r for b, r in reps.items() if b != a])
                   for a in alive}
        else:
            msgs = None if self.channel == "none" else self.inbox[team]
            out = {a: tx.describe(a, self.step_i, msgs) for a in alive}
        if self.history:
            for a, p in out.items():
                head, sep, tail = p.rpartition("Legal actions:")
                out[a] = head + self._history_block(team, a) + "\n" + sep + tail
        return out

    def _intent(self, text):
        if self.brief == "unified":
            return {"msg": parse_field(text, "msg", 8)}
        return {"focus": parse_field(text, "focus", 1), "plan": parse_field(text, "plan", 8)}

    # ---- per-unit history (unified protocol): last HIST_N steps of game facts, written by the environment ----
    HIST_N = 5

    @staticmethod
    def _act_name(act, n0):
        return {0: "no_op", 1: "stop", 2: "move_north", 3: "move_south", 4: "move_east", 5: "move_west"}.get(
            act, f"attack_e{act - n0}" if act >= n0 else str(act))

    def _record_history(self, team, acts, pre_own, pre_foe, cur_own, cur_foe, n0):
        h = self.hist[team]
        for a, act in enumerate(acts):
            if pre_own.get(a, 0) <= 0:
                continue
            name = self._act_name(act, n0)
            line = f"step {self.step_i - 1}: {name}"
            if act >= n0:
                k = act - n0
                line += f" -> e{k} killed" if pre_foe.get(k, 0) > 0 and cur_foe.get(k, 0) <= 0 else \
                        f" -> e{k} hp+shield {pre_foe.get(k, 0):.0f} -> {cur_foe.get(k, 0):.0f}"
            else:
                line += " ->"
            line += f"; you {pre_own[a]:.0f} -> {cur_own.get(a, 0):.0f}" + (" (died)" if cur_own.get(a, 0) <= 0 else "")
            h.setdefault(a, []).append(line)
            del h[a][:-self.HIST_N]

    def _history_block(self, team, a):
        lines = self.hist[team].get(a, [])
        score = self.score_smac[team]
        return ("Your last steps (oldest first):\n" + ("\n".join("  " + l for l in lines) if lines else "  (none yet)")
                + f"\nTeam score so far: {score:.1f} / 20")

    def _launch_b(self):
        """Team B decides on the current state in the background while team A's model thinks."""
        if self.tb is None:
            return
        prompts = self._prompts("B")
        self.prompts_b = prompts
        if self.pol_b is None:  # external: the caller sends team B's replies with step()
            return
        self.fut_b = self.pool.submit(self.pol_b.act, self.tx_b, list(prompts), prompts)

    def _view(self, reward=0.0, done=False, info=None):
        prompts = {} if done else {str(a): p for a, p in self._prompts("A").items()}
        out = {"step": self.step_i, "reward": reward, "done": done, "info": info or {}, "prompts": prompts}
        if STATE_OUT[0]:  # global state for a centralised critic (training only; never shown to agents)
            try:
                out["state"] = [round(float(x), 4) for x in self.env.get_state()]
            except Exception:
                out["state"] = None
        if self.opponent == "external":
            out["prompts_b"] = {} if done else {str(b): p for b, p in self._prompts("B").items()}
        return out

    # ------------------------------------------------------------------ API
    def reset(self, scenario: int | None = None):
        if scenario is not None:
            self._seed_scenario(int(scenario))
        self.runner.reset()
        self.step_i = 0
        self.inbox = {"A": [], "B": []}
        self.intents = {"A": {}, "B": {}}
        self.ret_b = 0.0
        self.cd_b, self._last_acts_b = {}, None
        self.hist = {"A": {}, "B": {}}
        self.score_smac = {"A": 0.0, "B": 0.0}
        self.b_log = []
        if self.shaper:
            self.shaper.reset(self.env)
            if self.tb is not None and self.shaper_b:
                self.shaper_b.reset(self.tb)
        self._launch_b()
        return self._view()

    def step(self, replies: dict[int, str], replies_b: dict[int, str] | None = None):
        env = self.env
        acts_a, valid, said = [], {}, {"A": [], "B": []}
        new_intents = {"A": {}, "B": {}}
        for a in range(env.n_agents):
            if a in replies and env.get_unit_by_id(a).health > 0:
                act, ok = self.tx_a.parse(a, replies[a])
                valid[str(a)] = ok
                if self.channel == "struct":
                    new_intents["A"][a] = self._intent(replies[a])
                say = parse_say(replies[a]) if self.channel != "none" else None
                if say:
                    said["A"].append((a, say))
            else:
                act = 0 if env.get_unit_by_id(a).health <= 0 else self.tx_a.default_action(a)
            acts_a.append(act)

        b_records = []
        if self.tb is not None:
            from .versus import _copy_hp, team_damage
            if self.pol_b is None:
                rep_b = {int(k): (v, None) for k, v in (replies_b or {}).items()}
            else:
                rep_b = self.fut_b.result()
            acts_b = []
            for b in range(self.tb.n_agents):
                if b in rep_b and self.tb.agents[b].health > 0:
                    text = rep_b[b][0]
                    act, ok = self.tx_b.parse(b, text)
                    say = parse_say(text) if self.channel != "none" else None
                    if say:
                        said["B"].append((b, say))
                    rec = {"agent": b, "prompt": self.prompts_b.get(b), "reply": text, "action": act,
                           "valid": ok, "say": say, "latency": rep_b[b][1]}
                    if self.channel == "struct":
                        new_intents["B"][b] = self._intent(text)
                        rec.update(new_intents["B"][b])
                    b_records.append(rec)
                else:
                    act = 0
                acts_b.append(act)
            self._last_acts_b = acts_b
            env._controller.pending_b = [c for b, act in enumerate(acts_b)
                                         if (c := self.tb.raw_action(b, act)) is not None]
            prev_a = {k: _copy_hp(u) for k, u in env.agents.items()}

        pre_a, pre_e = _hp(env.agents), _hp(env.enemies)
        reward, done, info = self.runner.step(acts_a)
        info = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in info.items()}
        self.step_i += 1
        self._update_cd_b()
        if self.history:
            n0 = env.n_actions_no_attack
            cur_a, cur_e = _hp(env.agents), _hp(env.enemies)
            self.score_smac["A"] += float(reward)
            self._record_history("A", acts_a, pre_a, pre_e, cur_a, cur_e, n0)
            if self.tb is not None:
                self._record_history("B", acts_b, pre_e, pre_a, cur_e, cur_a, n0)

        if self.tb is not None:
            max_r = env.max_reward / env.reward_scale_rate if env.reward_scale else 1.0
            r_b = team_damage(prev_a, env.agents, env.reward_death_value)
            a_alive = sum(u.health > 0 for u in env.agents.values())
            b_alive = sum(u.health > 0 for u in env.enemies.values())
            b_won = bool(done and a_alive == 0 and b_alive > 0)
            if b_won:
                r_b += env.reward_win
            self.ret_b += r_b / max_r
            self.score_smac["B"] = self.ret_b
            info.update(b_won=b_won, return_b=round(self.ret_b, 3), alive_a=a_alive, alive_b=b_alive)
            reward_b, own_b = r_b / max_r, None
            if self.shaper_b:
                info["reward_smac_b"] = float(reward_b)
                reward_b, own_b = self.shaper_b(self.tb, pre_e, pre_a, acts_b, bool(done), b_won,
                                                bool(info.get("battle_won")))
                info["reward_parts_b"] = {k: round(v, 4) for k, v in self.shaper_b.parts.items()}
            info["winner"] = "A" if info.get("battle_won") else ("B" if b_won else "draw") if done else None

        # deliver messages for the next step
        if self.channel == "team":
            self.inbox = {t: [f"a{a}: {s}" for a, s in said[t]] for t in ("A", "B")}
        elif self.channel == "all":  # own units as a<i>, the other team's as e<i> (matches unit ids)
            self.inbox = {
                "A": [f"a{a}: {s}" for a, s in said["A"]] + [f"e{b}: {s}" for b, s in said["B"]],
                "B": [f"a{b}: {s}" for b, s in said["B"]] + [f"e{a}: {s}" for a, s in said["A"]]}

        if self.msg_ablation == "drop":
            self.inbox = {"A": [], "B": []}
        elif self.msg_ablation == "shuffle":
            for t in self.inbox:
                txt = [m.split(": ", 1) for m in self.inbox[t]]
                senders = [x[0] for x in txt]
                random.shuffle(senders)
                self.inbox[t] = [f"{s_}: {x[1]}" for s_, x in zip(senders, txt)]
        info["messages"] = {t: list(v) for t, v in self.inbox.items()} if self.channel in ("team", "all") else None
        if self.channel == "struct":
            self.intents = new_intents
            info["intents"] = {t: {str(k): v for k, v in d.items()} for t, d in new_intents.items()}
        own = None
        if self.shaper:
            won = bool(info.get("battle_won"))
            lost = bool(done) and not won and all(u.health <= 0 for u in env.agents.values())
            info["reward_smac"] = float(reward)
            reward, own = self.shaper(env, pre_a, pre_e, acts_a, bool(done), won, lost)
            info["reward_parts"] = {k: round(v, 4) for k, v in self.shaper.parts.items()}
        out = self._view(float(reward), bool(done), info)
        out["valid"] = valid
        out["actions_a"] = {str(a): int(x) for a, x in enumerate(acts_a)}
        if own is not None:
            out["reward_unit"] = own
        if self.tb is not None:
            out["team_b"] = b_records
            out["reward_b"] = float(reward_b)
            out["valid_b"] = {str(r["agent"]): r["valid"] for r in b_records}
            if own_b is not None:
                out["reward_unit_b"] = own_b
        if not done:
            self._launch_b()
        return out

    def close(self):
        self.runner.close()
