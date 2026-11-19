"""Do team B's attack orders reach the game? A stands still; every B unit attacks A's unit 0 whenever legal."""
import json
from smac_llm.game import Game
from smac_llm.game import _hp
g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 11,
         opponent="external", channel="none", brief="unified", reward_mode="shaped2")
for sc in (3, 5):
    st = g.reset(scenario=sc)
    start = _hp(g.env.agents); legal = issued = 0
    while not st["done"] and st["step"] < 60:
        ra = {int(a): json.dumps({"action": "stop"}) for a in st["prompts"]}
        rb = {}
        for b in st["prompts_b"]:
            m = g.tx_b.action_menu(int(b))
            if "attack_e0" in m:
                rb[int(b)] = json.dumps({"action": "attack_e0"}); legal += 1
            else:
                mv = [k for k in m if k.startswith("move")]
                # walk toward A's unit 0
                u0, me = g.env.agents[0], g.tb.agents[int(b)]
                dx, dy = u0.pos.x - me.pos.x, u0.pos.y - me.pos.y
                d = ("move_east" if dx > 0 else "move_west") if abs(dx) > abs(dy) else ("move_north" if dy > 0 else "move_south")
                rb[int(b)] = json.dumps({"action": d if d in m else "stop"})
        st = g.step(ra, rb)
    end = _hp(g.env.agents)
    print(f"scenario {sc}: steps {st['step']} B attack_e0 orders {legal}; damage taken by A units:",
          {k: round(start[k] - end[k]) for k in start})
g.close()
