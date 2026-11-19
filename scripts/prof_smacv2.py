import time, random, sys
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
race, n = sys.argv[1], int(sys.argv[2])
units = {"terran": ["marine","marauder","medivac"], "protoss": ["stalker","zealot","colossus"], "zerg": ["zergling","hydralisk","baneling"]}[race]
cfg = {"n_units": n, "n_enemies": n,
       "team_gen": {"dist_type": "weighted_teams", "unit_types": units, "weights": [0.45,0.45,0.1],
                    "exception_unit_types": [units[2]] if race!="zerg" else ["baneling"], "observe": True},
       "start_positions": {"dist_type": "surrounded_and_reflect", "p": 0.5, "n_enemies": n, "map_x": 32, "map_y": 32}}
env = StarCraftCapabilityEnvWrapper(capability_config=cfg, map_name=f"10gen_{race}", conic_fov=False,
                                    use_unit_ranges=True, min_attack_range=2, obs_own_pos=True, fully_observable=False)
for ep in range(3):
    t=time.time(); env.reset(); print("reset", round(time.time()-t,2), flush=True)
    done=False; ts=[]
    while not done:
        acts=[]
        for a in range(env.env.n_agents):
            av=env.get_avail_agent_actions(a); acts.append(random.choice([i for i,x in enumerate(av) if x]))
        t=time.time(); r,done,info=env.step(acts); ts.append(time.time()-t)
    print("steps",len(ts),"mean step s",round(sum(ts)/len(ts),3), info, flush=True)
env.close()
