"""Text interface to SMAC/SMACv2 for LLM agents.

Each living agent gets a short text observation of what its own unit can see (partial
observability is kept: only units inside its sight range are listed) and a menu of the actions
that are legal right now. The model answers with one JSON object naming one menu item. Parsing
never crashes the game: an unreadable or illegal answer falls back to a legal default and is
recorded as invalid, so the invalid-action rate is a metric of its own.
"""
from __future__ import annotations

import json
import math
import re

MOVE_NAMES = {1: "stop", 2: "move_north", 3: "move_south", 4: "move_east", 5: "move_west"}
OWN_TYPE_ATTRS = ["marine", "marauder", "medivac", "stalker", "zealot", "colossus",
                  "zergling", "baneling", "hydralisk"]

SYSTEM_PROMPT = (
    "You control ONE unit in a StarCraft II team battle. Your team shares one goal: destroy all "
    "enemy units while losing as few of your own as possible. Every teammate is controlled by a "
    "separate copy of you that sees only what its own unit sees. North is +y, east is +x.\n"
    "Reply with ONLY a JSON object such as {\"action\": \"attack_e1\"}. The action must be one "
    "of the names listed under 'Legal actions'."
)


STRATEGY_PROMPT = """
GAME RULES
- One step = 0.5 game seconds. Your unit only shoots when you choose attack_eK. While you move or stop you do not shoot, even if you are being hit.
- Damage hits shields first. Once shields are 0, the rest hits health, reduced by 1 armor per hit. Health never regenerates; shields regenerate 2 per second after 10 seconds without damage.
- After firing, your weapon cools down ("weapon cooldown", in game loops; one step = 8 loops). At 0 you can fire right away.
UNITS (health/shield; damage; time between attacks; attack range; speed)
- zealot: 100/50; melee, 8 x2 per attack; 1.2 s; range 2 (must walk up to the target); speed 2.25.
- stalker: 80/80; 13, or 18 vs stalker and colossus; 1.87 s; range 6; speed 2.95 (fastest unit).
- colossus: 200/150; 10 x2, or 15 x2 vs zealot, and the beam also hits every enemy along a short line; 1.5 s; range 7; speed 2.25. Your most valuable unit.
ENEMY BEHAVIOUR
- Every enemy attacks the nearest of your units and keeps chasing that same unit until it dies. Enemies never focus fire and never retreat.
TACTICS
1. Focus fire: attack the enemy with the least health+shield that is in range, the same one as your teammates, and finish it before switching.
2. If your weapon cooldown is above 0 and an enemy is close, step away toward your teammates instead of standing still; attack again when the cooldown is 0.
3. Stalkers beat zealots by kiting: stay 3-6 away from them and shoot whenever the cooldown is 0.
4. A unit that is being chased should retreat behind its teammates, so the chaser walks into their fire.
5. Keep the colossus behind the stalkers; aim it at groups of zealots.
6. Stay together; never walk alone toward the enemy. Do not run away for the whole game: a draw at the time limit counts as a failure."""


FACTS_PROMPT = """
GAME RULES
- One step = 0.5 game seconds. Your unit only shoots when you choose attack_eK. While you move or stop you do not shoot, even if you are being hit.
- Damage hits shields first. Once shields are 0, the rest hits health, reduced by 1 armor per hit. Health never regenerates; shields regenerate 2 per second after 10 seconds without damage.
- After firing, your weapon cools down ("weapon cooldown", in game loops; one step = 8 loops). At 0 you can fire right away.
UNITS (health/shield; damage; time between attacks; attack range; speed)
- zealot: 100/50; melee, 8 x2 per attack; 1.2 s; range 2 (must walk up to the target); speed 2.25.
- stalker: 80/80; 13, or 18 vs stalker and colossus; 1.87 s; range 6; speed 2.95 (fastest unit).
- colossus: 200/150; 10 x2, or 15 x2 vs zealot, and the beam also hits every enemy along a short line; 1.5 s; range 7; speed 2.25. Your most valuable unit.
ENEMY BEHAVIOUR
- Every enemy attacks the nearest of your units and keeps chasing that same unit until it dies. Enemies never focus fire and never retreat.
YOUR TACTICS
You decide your own tactics; nothing below is required. Examples of what players sometimes do:
- several units attack the same weak enemy so it dies quickly;
- a unit steps back while its weapon cools down, then attacks again;
- a stalker keeps its distance from zealots and shoots them;
- a unit that is being chased moves behind its teammates.
Use an idea only when it fits the situation, and remember that a unit that moves does not shoot."""

RULES_PROMPT = FACTS_PROMPT.split("YOUR TACTICS")[0] + "YOUR TACTICS\nYou decide your own tactics."

# none: 5-line system prompt only | rules: rules + unit data + enemy behaviour, no tactics |
# facts: rules + optional tactic examples | tactics: rules + prescribed tactics
BRIEFS = {"none": "", "rules": RULES_PROMPT, "facts": FACTS_PROMPT, "tactics": STRATEGY_PROMPT}

STRUCT_SUFFIX = (
    "\nTEAM CHANNEL\nEvery step you receive 'Team reports': one JSON object per living teammate with its "
    "type, position, hp, shield, weapon cooldown, the enemies it can see (id, type, position, hp, shield), and "
    "the 'focus' and 'plan' it sent last step. Your own report is sent automatically.\n"
    "Talk, then fire together: add \"focus\": \"eK\" (the enemy you want the team to attack) whenever you "
    "attack or want others to; if a teammate already named a focus you can hit, consider joining it. Use "
    "\"plan\" (at most 8 words) to agree a team strategy, e.g. who advances, who holds back, when to regroup. "
    "Keep messages short. Example: {\"action\": \"attack_e2\", \"focus\": \"e2\", \"plan\": \"all on e2, then e4\"}.")

INFO_SUFFIX = (
    "\nTEAM CHANNEL\nEvery step you receive 'Team reports': one JSON object per living teammate with its "
    "type, position, hp, shield, weapon cooldown and the enemies it can see (id, type, position, hp, shield). "
    "Teammates cannot send other messages.")


def parse_field(text: str, key: str, max_words: int = 8) -> str | None:
    """Optional string field from a JSON reply, trimmed to max_words words / 80 chars."""
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        v = json.loads(m.group(0)).get(key)
    except (json.JSONDecodeError, AttributeError):
        m = re.search(r"\{.*?\}", text or "", re.S)
        try:
            v = json.loads(m.group(0)).get(key) if m else None
        except (json.JSONDecodeError, AttributeError):
            return None
    if not isinstance(v, str) or not v.strip():
        return None
    return " ".join(v.split()[:max_words])[:80]


COMM_SUFFIX = (" You may also add a \"say\" field (at most 12 words) that your teammates will read "
               "next step, e.g. {\"action\": \"attack_e1\", \"say\": \"focus e1, it is weak\"}.")


def parse_say(text: str) -> str | None:
    """Extract an optional team message from a reply (trimmed to 12 words / 100 chars)."""
    m = re.search(r"\{.*?\}", text or "", re.S)
    if not m:
        return None
    try:
        say = json.loads(m.group(0)).get("say")
    except (json.JSONDecodeError, AttributeError):
        return None
    if not isinstance(say, str) or not say.strip():
        return None
    return " ".join(say.split()[:12])[:100]


def _enemy_type_name(unit_type: int) -> str:
    from pysc2.lib import units as sc2_units  # lazy: the GPU learner does not need pysc2
    for race in (sc2_units.Terran, sc2_units.Protoss, sc2_units.Zerg):
        try:
            return race(unit_type).name.lower()
        except ValueError:
            continue
    return f"unit{unit_type}"


def snapshot(env, tenv) -> dict:
    """Positions/health of every unit, for replay rendering (not shown to the agents)."""
    def row(i, u, own):
        return {"id": i, "type": tenv._own_type_name(u.unit_type),
                "x": round(u.pos.x, 2), "y": round(u.pos.y, 2),
                "hp": round(u.health, 1), "hp_max": round(u.health_max, 1),
                "sh": round(getattr(u, "shield", 0), 1), "sh_max": round(getattr(u, "shield_max", 0), 1)}
    return {"allies": [row(i, u, True) for i, u in env.agents.items()],
            "enemies": [row(i, u, False) for i, u in env.enemies.items()]}


class TextSMAC:
    def __init__(self, env):
        self.env = env

    # ---- unit helpers -------------------------------------------------------------------
    def _own_type_name(self, unit_type: int) -> str:
        for name in OWN_TYPE_ATTRS:
            uid = getattr(self.env, f"{name}_id", 0)
            if uid and uid == unit_type:
                return name
        return _enemy_type_name(unit_type)

    @staticmethod
    def _hp(unit) -> str:
        s = f"hp {unit.health:.0f}/{unit.health_max:.0f}"
        if getattr(unit, "shield_max", 0):
            s += f", shield {unit.shield:.0f}/{unit.shield_max:.0f}"
        return s

    # ---- observation --------------------------------------------------------------------
    def action_menu(self, agent_id: int) -> dict[str, int]:
        avail = self.env.get_avail_agent_actions(agent_id)
        n_no_attack = self.env.n_actions_no_attack
        menu = {}
        for a, ok in enumerate(avail):
            if not ok:
                continue
            if a == 0:
                menu["no_op"] = 0
            elif a in MOVE_NAMES:
                menu[MOVE_NAMES[a]] = a
            elif a >= n_no_attack:
                menu[f"attack_e{a - n_no_attack}"] = a
        return menu

    def team_report(self, agent_id: int, intent: dict | None = None) -> dict:
        """Status report a unit broadcasts to its team (channel 'struct' / 'vision')."""
        env = self.env
        me = env.get_unit_by_id(agent_id)
        sight = env.unit_sight_range(agent_id)
        sees = [{"id": f"e{e}", "type": self._own_type_name(u.unit_type), "pos": [round(u.pos.x, 1), round(u.pos.y, 1)],
                 "hp": round(u.health), "shield": round(getattr(u, "shield", 0))}
                for e, u in env.enemies.items()
                if u.health > 0 and math.dist((me.pos.x, me.pos.y), (u.pos.x, u.pos.y)) < sight]
        r = {"from": f"a{agent_id}", "type": self._own_type_name(me.unit_type),
             "pos": [round(me.pos.x, 1), round(me.pos.y, 1)], "hp": round(me.health),
             "shield": round(getattr(me, "shield", 0)), "cooldown": round(me.weapon_cooldown), "sees": sees}
        for k, v in (intent or {}).items():
            if v:
                r[k] = v
        return r

    def describe(self, agent_id: int, step: int, messages: list[str] | None = None,
                 reports: list[dict] | None = None) -> str:
        """Text observation for one unit. `messages` (optional) are lines such as
        "a1: focus e2" from the previous step's message channel; `reports` (optional) are
        teammates' status reports (channel 'struct' / 'vision'), shown as JSON lines."""
        env = self.env
        me = env.get_unit_by_id(agent_id)
        sight = env.unit_sight_range(agent_id)
        lines = [f"Step {step}. You are agent a{agent_id}, a {self._own_type_name(me.unit_type)} "
                 f"at ({me.pos.x:.1f}, {me.pos.y:.1f}), {self._hp(me)}, "
                 f"weapon cooldown {me.weapon_cooldown:.0f}, shoot range "
                 f"{env.unit_shoot_range(agent_id):.0f}, sight range {sight:.0f}."]

        allies = []
        for a_id, u in env.agents.items():
            if a_id == agent_id or u.health <= 0:
                continue
            d = math.dist((me.pos.x, me.pos.y), (u.pos.x, u.pos.y))
            if d < sight:
                allies.append(f"  a{a_id} {self._own_type_name(u.unit_type)} at "
                              f"({u.pos.x:.1f}, {u.pos.y:.1f}), dist {d:.1f}, {self._hp(u)}")
        lines.append("Visible teammates:" + ("\n" + "\n".join(allies) if allies else " none"))

        menu = self.action_menu(agent_id)
        enemies = []
        for e_id, u in env.enemies.items():
            if u.health <= 0:
                continue
            d = math.dist((me.pos.x, me.pos.y), (u.pos.x, u.pos.y))
            if d < sight:
                tag = " [in range]" if f"attack_e{e_id}" in menu else ""
                enemies.append(f"  e{e_id} {self._own_type_name(u.unit_type)} at "
                               f"({u.pos.x:.1f}, {u.pos.y:.1f}), dist {d:.1f}, {self._hp(u)}{tag}")
        alive_e = sum(u.health > 0 for u in env.enemies.values())
        alive_a = sum(u.health > 0 for u in env.agents.values())
        lines.append("Visible enemies:" + ("\n" + "\n".join(enemies) if enemies else " none"))
        lines.append(f"Units alive: your team {alive_a}, enemy {alive_e}.")
        if messages is not None:
            lines.append("Messages from last step:" + ("\n" + "\n".join(f"  {m}" for m in messages)
                                                       if messages else " none"))
        if reports is not None:
            lines.append("Team reports:" + ("\n" + "\n".join("  " + json.dumps(r, separators=(",", ":"))
                                                            for r in reports) if reports else " none"))
        lines.append("Legal actions: " + ", ".join(menu))
        return "\n".join(lines)

    # ---- action parsing -----------------------------------------------------------------
    def parse(self, agent_id: int, text: str) -> tuple[int, bool]:
        """Map a model reply to a legal action id. Returns (action, was_valid)."""
        menu = self.action_menu(agent_id)
        name = None
        m = re.search(r"\{.*?\}", text or "", re.S)
        if m:
            try:
                name = str(json.loads(m.group(0)).get("action", "")).strip()
            except (json.JSONDecodeError, AttributeError):
                name = None
        if name is None:  # tolerate a bare action name
            m = re.search(r"(attack_e\d+|move_(?:north|south|east|west)|stop|no_op)", text or "")
            name = m.group(1) if m else None
        if name in menu:
            return menu[name], True
        m = re.fullmatch(r"attack_e(\d+)", name or "")
        if m and int(m.group(1)) in self.env.enemies:
            # out-of-range attack: close the distance to that target instead (still logged invalid)
            return self.move_toward(agent_id, self.env.enemies[int(m.group(1))]), False
        return self.default_action(agent_id), False

    def move_toward(self, agent_id: int, target) -> int:
        me = self.env.get_unit_by_id(agent_id)
        dx, dy = target.pos.x - me.pos.x, target.pos.y - me.pos.y
        prefs = (["move_east", "move_west"] if dx >= 0 else ["move_west", "move_east"])
        prefs_y = (["move_north", "move_south"] if dy >= 0 else ["move_south", "move_north"])
        order = [prefs[0], prefs_y[0]] if abs(dx) >= abs(dy) else [prefs_y[0], prefs[0]]
        menu = self.action_menu(agent_id)
        for name in order:
            if name in menu:
                return menu[name]
        return self.default_action(agent_id)

    def default_action(self, agent_id: int) -> int:
        """Fallback when a reply is illegal: attack the weakest enemy in range, else stop."""
        menu = self.action_menu(agent_id)
        attacks = [(self.env.enemies[int(k[8:])].health, v) for k, v in menu.items()
                   if k.startswith("attack_e")]
        if attacks:
            return min(attacks)[1]
        return menu.get("stop", menu.get("no_op", 0))



# ---- unified protocol (docs/unified_protocol.md, 2026-11-17): one system prompt for every condition ----
UNIFIED_PROMPT = """
GAME RULES
- One step = 0.5 game seconds. A game ends when one team has no units left, or after 200 steps (100 seconds); if both teams still have units then, it is a draw.
- Your unit only shoots when you choose attack_eK; attack_eK is listed only while that enemy is within your shoot range. While you move or stop you do not shoot, even if you are being hit.
- Damage hits shields first; shields have no armor. Once shields are 0, the rest hits health, reduced by 1 armor per hit. Health never regenerates; shields regenerate 2 per second after 10 seconds without taking damage.
- After firing, your weapon cools down ("weapon cooldown", in game loops; one step = 8 loops). At 0 you can fire right away.
UNITS (health/shield; damage per attack; time between attacks; attack range; speed)
- zealot: 100/50; melee, 8 x2; 1.2 s; range 2 (must walk up to the target); speed 2.25.
- stalker: 80/80; 13, or 18 vs stalker and colossus; 1.87 s; range 6; speed 2.95.
- colossus: 200/150; 10 x2, or 15 x2 vs zealot, and the beam also hits every enemy along a short line; 1.5 s; range 7; speed 2.25.
ENEMY
- The enemy team has the same unit types as your team and follows the same rules.
TACTICS
- You decide your own tactics.
YOUR HISTORY
- Every step you also see your own last 5 steps: the action you took, what happened to your target and to your own hp+shield, and your team's game score so far (the game's default score, 0 to 20)."""

UNIFIED_INFO = (
    "\nTEAM REPORTS\nEvery step you receive one JSON object per living teammate: its type, position, hp, shield, "
    "weapon cooldown and the enemies it can see (id, type, position, hp, shield). Your own report is sent automatically.")
UNIFIED_MSG = (
    "\nTEAM MESSAGES\nYou may add a \"msg\" field (at most 8 words) to your reply; your teammates see it in your "
    "next report. Format: {\"action\": \"<one legal action>\", \"msg\": \"<at most 8 words>\"}")
UNIFIED_CHANNEL = {"none": "", "info": UNIFIED_INFO, "struct": UNIFIED_INFO + UNIFIED_MSG}
BRIEFS["unified"] = UNIFIED_PROMPT

VISION_SUFFIX = INFO_SUFFIX  # old name
