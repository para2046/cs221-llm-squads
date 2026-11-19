"""CPU check of smac_llm.game: scenario replay, opponents, and the message channel."""
import json
import random
import sys

from smac_llm.game import Game

env_args = dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None)


def units(g):
    return sorted((g.tx_a._own_type_name(u.unit_type), round(u.pos.x, 1), round(u.pos.y, 1))
                  for u in g.env.agents.values())


def play(g, scenario, say=False):
    v = g.reset(scenario)
    first = units(g)
    while not v["done"]:
        reps = {}
        for a, p in v["prompts"].items():
            acts = [x for x in p.split("Legal actions: ")[1].split(", ")]
            r = {"action": random.choice(acts)}
            if say:
                r["say"] = f"unit {a} going {r['action']}"
            reps[int(a)] = json.dumps(r)
        v = g.step(reps)
    return first, v["info"], v


opp = sys.argv[1] if len(sys.argv) > 1 else "builtin"
chan = sys.argv[2] if len(sys.argv) > 2 else "none"
g = Game(env_args, seed=3, opponent=opp, channel=chan, incentive_b=None)
a1, i1, _ = play(g, 42, say=chan != "none")
a2, i2, _ = play(g, 42, say=chan != "none")
a3, i3, _ = play(g, 7, say=chan != "none")
print("same scenario -> same units:", a1 == a2, "| different scenario differs:", a1 != a3)
print("info:", i1)
v = g.reset(5)
v = g.step({int(a): json.dumps({"action": "stop", "say": "hold here, wait for e2"}) for a in v["prompts"]})
print("--- prompt after one step with messages ---")
print(next(iter(v["prompts"].values())))
if v.get("team_b"):
    print("team_b record:", v["team_b"][0]["reply"])
g.close()
