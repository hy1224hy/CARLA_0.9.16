"""Feasibility, private-information and single/dual scripted attack controls."""
from dataclasses import replace
import json
import numpy as np
from .evaluation import evaluate
from .environment import BottleneckEnv
from ..training.checkpoint import source_digest

def calibrate(config,per_family=8,capture=False):
    t=replace(config.task,attack_weight=1.)
    results={};preview=[]
    for mode in ('normal','A','B','both'):
        print(f'calibrating bottleneck script={mode}',flush=True)
        report,frames=evaluate(None,t,'gate',per_family,gamma=config.gamma,script=mode,capture=capture and mode=='both')
        results[mode]=report
        if frames:preview=frames
    a,b=BottleneckEnv(t),BottleneckEnv(t)
    try:
        oa,ia=a.reset(seed=123,options={'family':'lead_merge','red_target':16.})
        ob,ib=b.reset(seed=123,options={'family':'lead_merge','red_target':20.})
        info={'blue_A_observation_changes':bool(np.max(abs(oa[0]-ob[0]))>0),
              'blue_B_observation_identical':bool(np.array_equal(oa[1],ob[1])),
              'communication_available':bool(ia['links'].all() and ib['links'].all())}
    finally:a.close();b.close()
    reference_ok=all(r['overall']['baseline_valid_rate']==1 for r in results.values())
    normal=results['normal']['overall']
    positive=results['both']['overall']['qualified_attack_utility']>0
    passed=bool(reference_ok and normal['success_rate']==1 and normal['bad_rate']==0
                and abs(normal['qualified_attack_utility'])<1e-8 and positive and all(info.values()))
    scores={k:r['overall']['qualified_attack_utility'] for k,r in results.items()}
    receipt={'passed':passed,'source_sha256':source_digest(),'config_sha256':config.digest(),
        'split':'gate','cases_per_family':per_family,'information_check':info,'script_U':scores,
        'dual_exceeds_best_single':scores['both']>max(scores['A'],scores['B']),
        'claims':'feasibility controls only; no claim learned communication or algorithm superiority',
        'reports':results}
    return receipt,preview
