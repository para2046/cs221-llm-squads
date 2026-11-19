"""Unit tests for ShapedReward2 per-unit credit: sum over live units is 0, and D equals the live total,
in death / splash (unattributed) / shield-regeneration / all-dead cases. Run: python tests/test_reward_v2.py"""
import sys, types, random
sys.path.insert(0, ".")
sys.modules.setdefault("numpy", __import__("numpy"))
from smac_llm.game import ShapedReward2, _hp

class U:
    def __init__(s, h, sh, hm=100, shm=50): s.health, s.shield, s.health_max, s.shield_max = h, sh, hm, shm
class Env:
    n_actions_no_attack = 6
    def __init__(s, a, e): s.agents, s.enemies = a, e

def run(prev_a, prev_e, cur_a, cur_e, acts, done=False, won=False, lost=False):
    env = Env({k: U(*v) for k, v in prev_a.items()}, {k: U(*v) for k, v in prev_e.items()})
    r = ShapedReward2(); r.reset(env)
    pa, pe = _hp(env.agents), _hp(env.enemies)
    env.agents = {k: U(*v) for k, v in cur_a.items()}; env.enemies = {k: U(*v) for k, v in cur_e.items()}
    team, own = r(env, pa, pe, acts, done, won, lost)
    return team, own, pa

full = {i: (100, 50) for i in range(5)}
cases = {
 "plain hits":  (full, full, {0: (90, 50), 1: (100, 50), 2: (100, 30), 3: (100, 50), 4: (100, 50)},
                 {0: (100, 0), 1: (60, 0), 2: (100, 50), 3: (100, 50), 4: (100, 50)}, [6, 7, 7, 1, 2]),
 "pure own death (team -0.15)": (full, full, {**full, 0: (0, 0)}, full, [1, 1, 1, 1, 1]),
 "own death + enemy kill (trade, team 0)": (full, full, {0: (0, 0), 1: (100, 50), 2: (100, 50), 3: (100, 50), 4: (100, 50)},
                 {0: (0, 0), 1: (100, 50), 2: (100, 50), 3: (100, 50), 4: (100, 50)}, [6, 6, 6, 1, 1]),
 "splash (enemy hit by nobody targeting it)": (full, full, full,
                 {0: (100, 20), 1: (100, 20), 2: (100, 50), 3: (100, 50), 4: (100, 50)}, [6, 1, 1, 1, 1]),
 "shield regen": ({**full, 1: (100, 10)}, full, {**full, 1: (100, 14)}, full, [1, 1, 1, 1, 1]),
 "already dead units": ({**full, 3: (0, 0), 4: (0, 0)}, full, {**full, 3: (0, 0), 4: (0, 0), 0: (50, 0)},
                 {**full, 2: (0, 0)}, [8, 8, 8, 0, 0]),
}
ok = True
for name, (pa_, pe_, ca, ce, acts) in cases.items():
    team, own, pa = run(pa_, pe_, ca, ce, acts)
    live = [a for a in range(5) if pa[a] > 0]
    s = sum(own.values())
    good = abs(s) < 1e-9 and set(own) == {str(a) for a in live}
    ok &= good
    print(f"{name:45s} team={team:+.4f} sum(own)={s:+.2e} live={live} {'OK' if good else 'FAIL'}")
random.seed(0)
for _ in range(2000):   # random states
    pa_ = {i: (random.choice([0, random.uniform(1, 100)]), random.uniform(0, 50)) for i in range(5)}
    pe_ = {i: (random.choice([0, random.uniform(1, 100)]), random.uniform(0, 50)) for i in range(5)}
    ca = {i: ((max(0, v[0] - random.uniform(0, 60)) if v[0] > 0 else 0), random.uniform(0, 50)) for i, v in pa_.items()}
    ce = {i: ((max(0, v[0] - random.uniform(0, 60)) if v[0] > 0 else 0), random.uniform(0, 50)) for i, v in pe_.items()}
    acts = [random.randrange(11) for _ in range(5)]
    team, own, pa = run(pa_, pe_, ca, ce, acts)
    if own and abs(sum(own.values())) > 1e-9:
        ok = False; print("FAIL random", sum(own.values()))
print("random 2000 states:", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
