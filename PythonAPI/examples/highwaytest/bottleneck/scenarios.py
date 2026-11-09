import numpy as np
FAMILIES=('lead_merge','rear_merge')
OFFSETS={'train':0,'gate':4000000,'selection':5000000,'test':6000000}

def cases(split,per_family):
    if type(per_family) is not int or not 1<=per_family<=100000:
        raise ValueError('invalid bottleneck case count')
    return [(f,OFFSETS[split]+i*100000+j) for i,f in enumerate(FAMILIES) for j in range(per_family)]

def starts(rng,family,t,red_target=None):
    if family not in FAMILIES:raise ValueError('unknown bottleneck family')
    red_x=float(60+rng.uniform(-3,3));a=red_x+float(rng.uniform(18,24))
    # Keep the rear participant within radio range, but avoid a closing pair
    # reaching overlapping taper lanes simultaneously under the normal IDM.
    gap=float(rng.uniform(34,44) if family=='lead_merge' else rng.uniform(60,70))
    b=a+gap*(1 if family=='lead_merge' else -1)
    sampled_target=float(t.red_target_speed+rng.choice([-2.,2.]))
    target=sampled_target if red_target is None else float(red_target)
    return {'family':family,'red_target_speed':target,
            'layout':[(red_x,0.,max(8.,target-1)),(a,0.,14.),(b,4.,16.)]}
