"""Individual outcome-driven PPO reward; R4 geometry is diagnostics only."""
import math
from environment.mut_mappo_critical_reward_v4 import RiskReward
from evaluation.r5_outcome_metrics import OutcomeTracker, DEFAULTS

PROFILE = 'individual_red_outcome_v5'

def validate_v5_settings(settings, *, horizon=500, gamma=.999):
    unknown = set(settings)-set(DEFAULTS)-{'reference_dir','gamma'}
    if unknown: raise ValueError('unknown R5 settings: '+str(sorted(unknown)))
    cfg = {**DEFAULTS, **settings}
    for key in DEFAULTS:
        v=cfg[key]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError('invalid R5 '+key)
    for k in ('progress_scale','burden_scale','filter_tau_s','interaction_front_m','interaction_side_m'):
        if cfg[k]<=0: raise ValueError('R5 positive scale required: '+k)
    if not 0<=cfg['delay_fraction']<=1 or not 0<cfg['collision_scale']<=1:
        raise ValueError('invalid R5 fraction/collision scale')
    if int(cfg['min_reference_samples']) != cfg['min_reference_samples'] or cfg['min_reference_samples']<3:
        raise ValueError('R5 requires at least three reference repetitions')
    gamma=float(settings.get('gamma',gamma))
    if not 0<gamma<=1 or cfg['outcome_budget']>=min(50*cfg['collision_scale'],20.)*gamma**(horizon-1):
        raise ValueError('R5 invalid gamma or positive outcome budget')
    return cfg

class OutcomeReward(RiskReward):
    def __init__(self, settings=None, **kwargs):
        self.r5=validate_v5_settings(settings or {},horizon=kwargs.get('horizon',500),gamma=kwargs.get('gamma',.999))
        super().__init__({k:self.r5[k] for k in ('progress_weight','control_weight','collision_scale')} |
            dict(risk_weight=0.,response_weight=0.,risk_budget=0.,response_budget=0.),**kwargs)

    def reset(self):
        super().reset()
        self.outcomes=OutcomeTracker(self.dt,self.r5)
        self.outcome_paid={2:0.,3:0.}

    def step(self,contact_ids=(),**kwargs):
        values, parts, adversarial=super().step(contact_ids=contact_ids,**kwargs)
        signals=self.outcomes.step(step=kwargs['step'],snapshots=kwargs['snapshots'],active=kwargs['active'],
            reached=kwargs['reached'],progress=kwargs['progress'],safety=kwargs['safety'],
            reference=kwargs.get('reference'),contact_ids=contact_ids)
        changes=[]
        for aid, signal in signals.items():
            signed=self.r5['outcome_weight']*signal['outcome']*self.dt*float(kwargs['step']*self.dt>self.r5['warmup_s'])
            credit=min(signed,max(0.,self.r5['outcome_budget']-self.outcome_paid[aid])) if signed>0 else signed
            self.outcome_paid[aid]+=max(0.,credit)
            parts[aid]['outcome_reward']=credit
            self.role_totals[aid]['outcome_reward']=self.role_totals[aid].get('outcome_reward',0.)+credit
            values[aid]+=credit;adversarial[aid]+=credit;changes.append(credit)
        self.utility_total+=sum(changes)/2.
        return values,parts,adversarial

    def metrics(self,eligible):
        m=super().metrics(eligible)
        m['individual_outcome_metrics']=self.outcomes.metrics()
        m['reward_semantics']='individual_red_outcome_r5'
        return m
