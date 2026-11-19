"""Render the unified system prompts and a mid-game observation with per-unit history (self-play and vs built-in)."""
import json, random, re
from smac_llm.game import Game, system_prompt
for ch in ("none", "info", "struct"):
    s = system_prompt(ch, None, "unified", builtin_enemy=True)
    assert s == system_prompt(ch, None, "unified", builtin_enemy=False)
    print(f"===== system prompt [{ch}] ({len(s.split())} words)\n{s}\n" if ch == "struct" else f"===== [{ch}] {len(s.split())} words")
def pick(m, prompt):
    at = [k for k in m if k.startswith("attack")]
    if at and random.random() < 0.8:
        return random.choice(at)
    x, y = map(float, re.search(r"at \(([\d.]+), ([\d.]+)\)", prompt).groups())
    mv = ("move_west" if x > 16 else "move_east") if abs(x - 16) > abs(y - 16) else ("move_south" if y > 16 else "move_north")
    return mv if mv in m else "stop"
for opp in ("external", "builtin"):
    g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 11,
             opponent=opp, channel="struct", brief="unified", reward_mode="shaped2" if opp == "external" else "smac")
    st = g.reset(scenario=3); shown = False
    while not st["done"]:
        ra = {int(a): json.dumps({"action": pick(g.tx_a.action_menu(int(a)), p), "msg": "test msg"}) for a, p in st["prompts"].items()}
        rb = {int(b): json.dumps({"action": pick(g.tx_b.action_menu(int(b)), p)}) for b, p in st.get("prompts_b", {}).items()} if opp == "external" else None
        st = g.step(ra, rb)
        if not shown and st["step"] >= 8 and st["prompts"] and any("killed" in p or "->" in p for p in st["prompts"].values()):
            k = sorted(st["prompts"])[0]
            print(f"===== [{opp}] team A unit {k} at step {st['step']}\n{st['prompts'][k]}\n")
            if opp == "external":
                kb = sorted(st["prompts_b"])[0]
                print(f"===== [{opp}] team B unit {kb}\n{st['prompts_b'][kb][-600:]}\n")
            shown = True
    print(f"[{opp}] end step {st['step']} winner {st['info'].get('winner')} scoreA {g.score_smac['A']:.2f} scoreB {g.score_smac['B']:.2f}")
    g.close()
