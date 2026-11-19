import time, random
from smacv2.env import StarCraft2Env
t=time.time(); env=StarCraft2Env(map_name="3m"); 
for ep in range(2):
    t=time.time(); env.reset(); print("reset", round(time.time()-t,2), flush=True)
    done=False; ts=[]
    while not done:
        acts=[]
        for a in range(env.n_agents):
            av=env.get_avail_agent_actions(a); acts.append(random.choice([i for i,x in enumerate(av) if x]))
        t=time.time(); r,done,info=env.step(acts); ts.append(time.time()-t)
    print("steps",len(ts),"mean step s",round(sum(ts)/len(ts),3), "max", round(max(ts),2), flush=True)
env.close()
