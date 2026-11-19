"""CPU check of reward v2 in self-play: team rewards are zero-sum and per-unit credits sum to zero."""
import json, random
from smac_llm.game import Game
g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 11,
         opponent="external", channel="struct", brief="rules", reward_mode="shaped2")
import re
def pick(m, prompt):
    at = [k for k in m if k.startswith("attack")]
    if at and random.random() < 0.8:
        return random.choice(at)
    x, y = map(float, re.search(r"at \(([\d.]+), ([\d.]+)\)", prompt).groups())
    mv = ("move_west" if x > 16 else "move_east") if abs(x - 16) > abs(y - 16) else ("move_south" if y > 16 else "move_north")
    return mv if mv in m else "stop"
for ep in range(2):
    st = g.reset(scenario=ep); ra = rb = 0.0; worst = worst_own = 0.0
    while not st["done"]:
        a = {int(x): json.dumps({"action": pick(g.tx_a.action_menu(int(x)), p)}) for x, p in st["prompts"].items()}
        b = {int(x): json.dumps({"action": pick(g.tx_b.action_menu(int(x)), p)}) for x, p in st["prompts_b"].items()}
        st = g.step(a, b); ra += st["reward"]; rb += st["reward_b"]
        worst = max(worst, abs(st["reward"] + st["reward_b"]))
        for k in ("reward_unit", "reward_unit_b"):
            worst_own = max(worst_own, abs(sum((st.get(k) or {}).values())))
    i = st["info"]
    print(f"ep{ep} steps={st['step']} winner={i.get('winner')} A={ra:.3f} B={rb:.3f} max|rA+rB|={worst:.2e} "
          f"max|sum own|={worst_own:.2e} partsA={i['reward_parts']} partsB={i['reward_parts_b']}")
g.close()
