"""CPU check of Game(opponent='external'): random replies for both teams, shaped rewards for both."""
import json, random
from smac_llm.game import Game
g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 11,
         opponent="external", channel="struct", brief="rules", reward_mode="shaped")
assert "another AI player" in g.system_a and "another AI player" in g.system_b
for ep in range(2):
    st = g.reset(scenario=ep); ra = rb = 0.0
    while not st["done"]:
        ra_ = {int(a): json.dumps({"action": random.choice(list(g.tx_a.action_menu(int(a)))), "focus": "e1"}) for a in st["prompts"]}
        rb_ = {int(b): json.dumps({"action": random.choice(list(g.tx_b.action_menu(int(b))))}) for b in st["prompts_b"]}
        st = g.step(ra_, rb_); ra += st["reward"]; rb += st["reward_b"]
    i = st["info"]
    print(f"ep{ep} steps={st['step']} winner={i['winner']} shaped A={ra:.2f} B={rb:.2f} partsA={i['reward_parts']} partsB={i['reward_parts_b']} validB={list(st['valid_b'].values())[:3]}")
g.close()
