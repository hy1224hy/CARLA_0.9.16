"""Versioned second candidate, based on the completed R0/R1 engineering pilot.

Primary event credit and stronger existing normal-driving costs. Qualification,
observations, actions, the legacy task-stall grace and red policy stay fixed.
"""
import math
from environment.mut_mappo_critical_reward import CriticalTeamReward, validate_settings
from evaluation.critical_interaction_metrics import obb_ttc, _box

PROFILE='normal_driving_critical_team_v2'
EXTRA_DEFAULTS=dict(primary_event_reward=5.,critical_credit_grace_s=1.,
    normal_overspeed_scale=10.,normal_comfort_scale=5.,normal_lane_scale=10.)


def validate_v2_settings(settings, *, horizon=500, gamma=.999):
    extra={k:settings.get(k,v) for k,v in EXTRA_DEFAULTS.items()}
    for key,value in extra.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('invalid normal-driving critical setting: '+key)
    if min(extra['normal_overspeed_scale'],extra['normal_comfort_scale'])<1:
        raise ValueError('normal-driving scales cannot refund existing safety costs')
    base=validate_settings({'critical_weight':3.,'pace_weight':.1,
        **{k:v for k,v in settings.items() if k not in EXTRA_DEFAULTS}})
    positive_bound=base['critical_weight']+extra['primary_event_reward']+base.get('attack_weight',2.)+base.get('driving_weight',1.)
    minimum_cost=min(base.get(k,d) for k,d in (('collision_cost',50.),('offroad_cost',50.),
        ('road_rule_cost',50.),('task_violation_cost',20.)))
    if positive_bound>=minimum_cost*gamma**(horizon-1):
        raise ValueError('positive episode budget must remain below the latest discounted safety event cost')
    return {**base,**extra}


class NormalDrivingCriticalReward(CriticalTeamReward):
    def __init__(self,settings=None,**kwargs):
        resolved=validate_v2_settings(settings or {},horizon=kwargs.get('horizon',500),gamma=kwargs.get('gamma',.999))
        self.normal={k:resolved[k] for k in EXTRA_DEFAULTS}
        super().__init__({k:v for k,v in resolved.items() if k not in EXTRA_DEFAULTS},**kwargs)

    def reset(self):
        super().reset()
        self.primary_event_paid=False
        self.primary_runs={}

    def step(self,**kwargs):
        kw=dict(kwargs)
        safety=kw['safety']
        safe=bool(kw['eligible']) and not self.seen_bad and not any(any(v.values()) for v in safety.values())
        # Credit warm-up is separate from legacy slowdown and stall warm-up.
        kw['eligible']=safe and kw['step']*self.dt>self.normal['critical_credit_grace_s']+1e-9
        utility,components=super().step(**kw)
        live=[i for i,on in kw['active'].items() if on]
        pairs=[]
        for rid in (0,1):
            for bid in (2,3):
                if rid in live and bid in live:
                    r,b=kw['snapshots'][rid],kw['snapshots'][bid]
                    pairs.append((rid,bid,r,b,obb_ttc(r,b,self.critical['prediction_horizon_s'])))
        severe=any(t is not None and t<=self.critical['severe_ttc_s'] for _,_,_,_,t in pairs)
        next_runs={}
        for rid,bid,red,blue,ttc in pairs:
            rc,axes,_,_=_box(red);bc,_,_,_=_box(blue)
            longitudinal=sum((bc[j]-rc[j])*axes[0][j] for j in (0,1))
            allowed=(kw['eligible'] and not severe and not red.get('reached',False)
                and kw['driving_context'][bid].get('legal_forward') is True and longitudinal>=0
                and ttc is not None and self.critical['severe_ttc_s']<ttc<=self.critical['event_ttc_s'])
            next_runs[(rid,bid)]=self.primary_runs.get((rid,bid),0)+1 if allowed else 0
        self.primary_runs=next_runs
        bonus=0.
        if not self.primary_event_paid and any(n*self.dt+1e-9>=self.critical['event_duration_s'] for n in next_runs.values()):
            bonus=self.normal['primary_event_reward'];self.primary_event_paid=True
        overspeed=(self.normal['normal_overspeed_scale']-1)*sum(c.get('overspeed',0.) for c in kw['costs'].values())/2
        comfort=(self.normal['normal_comfort_scale']-1)*max(0.,-components['comfort'])
        lane=0.
        for bid in (2,3):
            if bid not in live or self.normal['normal_lane_scale']==0:continue
            detail=kw['base_details'].get(bid,{})
            if detail.get('inactive'):continue
            if 'lane_deviation' not in detail:
                raise ValueError('active blue lacks the existing legally-exempted lane cost')
            lane+=self.normal['normal_lane_scale']*max(0.,-float(detail['lane_deviation']))/2
        extra=dict(primary_event_bonus=bonus,normal_overspeed=-overspeed,
                   normal_comfort=-comfort,normal_lane=-lane)
        delta=math.fsum(extra.values())
        if not math.isfinite(delta):raise FloatingPointError('nonfinite normal-driving critical reward')
        self.utility_total+=delta;self.cost_total+=overspeed+comfort+lane
        for key,value in extra.items():self.components_total[key]=self.components_total.get(key,0.)+value
        return utility+delta,{**components,**extra}

    def metrics(self,eligible):
        return {**super().metrics(eligible),
            'credited_primary_event':bool(self.primary_event_paid),
            'qualified_primary_event_credit':float(eligible and self.primary_event_paid),
            'critical_credit_grace_s':self.normal['critical_credit_grace_s']}
