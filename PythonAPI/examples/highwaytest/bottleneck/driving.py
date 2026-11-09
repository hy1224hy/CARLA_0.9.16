"""IDM recovery demonstrations in the new road; no old checkpoints or test cases."""
import numpy as np
from .evaluation import Backend
from .contract import inputs
from .scenarios import FAMILIES

def demonstrations(config,episodes):
    rng=np.random.default_rng(config.seed+52013)
    obs,actions,seeds,families=[],[],[],[]
    for ep in range(episodes):
        family=FAMILIES[ep%len(FAMILIES)]
        seed=int(rng.integers(0,900000));seeds.append(seed);families.append(family)
        b=Backend(config.task,'C1',family,seed,gamma=config.gamma)
        try:
            for v in b.road.vehicles[1:]:
                v.position[1]+=float(rng.uniform(-.4,.4));v.heading+=float(rng.uniform(-.05,.05))
                v.speed=float(rng.uniform(8,20))
            b.frame=b.env.snapshot(b.road,time=0)
            b.tracker=b.env.new_tracker();b.tracker.update(b.frame)
            while b.frame.time<config.task.duration-1e-6 and not b.tracker.bad:
                o,_,_=inputs(b.frame,config.task);b.road.act()
                a=np.asarray([[v.action['acceleration']/5,v.action['steering']/.25] for v in b.road.vehicles[1:]],np.float32)
                obs.append(o);actions.append(np.clip(a,-1,1));b.step()
        finally:b.close()
        if (ep+1)%16==0:print(f'bottleneck demonstrations {ep+1}/{episodes}',flush=True)
    return {'obs':np.asarray(obs,np.float32),'actions':np.asarray(actions,np.float32),
            'seeds':np.asarray(seeds,np.int64),'families':np.asarray(families)}
