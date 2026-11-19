"""CPU check of the shaped reward: random replies vs the built-in AI; parts must sum to the shaped return."""
import json, random
from smac_llm.game import Game
g = Game(dict(map="10gen_protoss", n_units=5, n_enemies=None, difficulty="7", replay_dir=None), 7, reward_mode="shaped")
for ep in range(4):
    st = g.reset(scenario=ep); R = Rs = 0.0; focus = 0.0
    while not st["done"]:
        reps = {}
        for a in st["prompts"]:
            menu = g.tx_a.action_menu(int(a)); reps[int(a)] = json.dumps({"action": random.choice(list(menu))})
        st = g.step(reps); R += st["reward"]; Rs += st["info"]["reward_smac"]; focus += sum(st.get("reward_unit", {}).values())
    parts = st["info"]["reward_parts"]
    team = sum(v for k, v in parts.items() if k != "focus")
    print(f"ep{ep} steps={st['step']} won={st['info'].get('battle_won')} smac_return={Rs:.2f} shaped={R:.3f} "
          f"parts={parts} team_sum_ok={abs(team - R) < 1e-6} focus_ok={abs(focus - parts['focus']) < 1e-6}")
g.close()
