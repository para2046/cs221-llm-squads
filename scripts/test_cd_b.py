import json, random, re, collections
from smac_llm.game import Game
g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 11,
         opponent="external", channel="struct", brief="rules", reward_mode="shaped")
c = collections.Counter()
def pick(m, mv):
    at = [k for k in m if k.startswith('attack')]
    return at[0] if at else (mv if mv in m else 'stop')
st = g.reset(scenario=0)
while not st["done"]:
    for p in st["prompts_b"].values():
        c[int(re.search(r"weapon cooldown (\d+)", p).group(1)) > 0] += 1
    ra = {int(a): json.dumps({"action": pick(g.tx_a.action_menu(int(a)), "move_east")}) for a in st["prompts"]}
    rb = {int(b): json.dumps({"action": pick(g.tx_b.action_menu(int(b)), "move_west")}) for b in st["prompts_b"]}
    st = g.step(ra, rb)
print("B prompts with cooldown>0:", c, "steps", st["step"], st["info"]["winner"])
g.close()
