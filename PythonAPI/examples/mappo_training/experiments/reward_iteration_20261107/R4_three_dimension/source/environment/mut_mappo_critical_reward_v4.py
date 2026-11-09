"""R4: individual uncapped risk diagnostics with bounded training credit."""
import math
from environment.base_reward_v6 import REWARD_COMPONENTS
from environment.mut_mappo_critical_reward_v3 import IndividualDrivingCriticalReward, HARD_COMPONENTS
from environment.mut_mappo_critical_reward_v2 import NormalDrivingCriticalReward
from evaluation.r4_risk_metrics import RiskTracker

PROFILE='individual_three_dimension_risk_v4'
DEFAULTS=dict(progress_weight=.03,control_weight=.3,risk_weight=1.,response_weight=.5,
              risk_budget=8.,response_budget=2.,collision_scale=1.)


def validate_v4_settings(settings,**kwargs):
    known=set(DEFAULTS)|{'reference_dir','gamma'}
    if set(settings)-known:raise ValueError('unknown R4 reward settings')
    resolved={**DEFAULTS,**settings}
    for k in DEFAULTS:
        v=resolved[k]
        if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:
            raise ValueError('invalid R4 '+k)
    if not 0<resolved['collision_scale']<=1:raise ValueError('R4 collision scale must be in (0,1]')
    gamma=float(settings.get('gamma',kwargs.get('gamma',.999)))
    if not 0<gamma<=1:raise ValueError('invalid R4 gamma')
    minimum=min(50.*resolved['collision_scale'],20.)
    if resolved['risk_budget']+resolved['response_budget']>=minimum*gamma**(kwargs.get('horizon',500)-1):
        raise ValueError('R4 positive budget exceeds discounted hard-cost budget')
    return resolved


class RiskReward(IndividualDrivingCriticalReward):
    def __init__(self,settings=None,**kwargs):
        self.r4=validate_v4_settings(settings or {},horizon=kwargs.get("horizon",500),gamma=kwargs.get("gamma",.999))
        # Ancestor is used only for existing team diagnostic fields/ABI.
        NormalDrivingCriticalReward.__init__(self,dict(attack_weight=0.,driving_weight=0.,
            pace_weight=0.,normal_lane_scale=0.,normal_comfort_scale=1.,normal_overspeed_scale=1.),**kwargs)

    def reset(self):
        NormalDrivingCriticalReward.reset(self)
        self.tracker=RiskTracker(self.dt);self.role_totals={2:{},3:{}}
        self.budgets={i:dict(risk=0.,response=0.) for i in (2,3)};self.task_paid=set()

    def step(self,contact_ids=(),**kwargs):
        _,geometry=NormalDrivingCriticalReward.step(self,**kwargs)
        signals=self.tracker.step(kwargs['step'],kwargs['snapshots'],kwargs['active'],kwargs['safety'],
            (kwargs.get('reference') or {}).get('response_states'),contact_ids)
        rewards={};details={};adversarial={}
        for aid,signal in signals.items():
            base=kwargs['base_details'][aid]
            if set(REWARD_COMPONENTS)-set(base):raise ValueError('R4 requires complete frozen v6 base components')
            mission=('progress','success','mission_timeout','free_flow_pace_deficit')
            soft=self.r4['progress_weight']*sum(base.get(k,0.) for k in mission)
            controls=self.r4['control_weight']*sum(base[k] for k in REWARD_COMPONENTS if k not in HARD_COMPONENTS and k not in mission)
            risk=min(self.r4['risk_weight']*signal['risk']*self.dt*float(kwargs['step']*self.dt>1.),max(0.,self.r4['risk_budget']-self.budgets[aid]['risk']))
            response=min(self.r4['response_weight']*signal['response_excess']*self.dt*float(kwargs['step']*self.dt>1.),max(0.,self.r4['response_budget']-self.budgets[aid]['response']))
            self.budgets[aid]['risk']+=risk;self.budgets[aid]['response']+=response
            task=kwargs['safety'].get(aid,{}).get('task_violation',False) and aid not in self.task_paid
            if task:self.task_paid.add(aid)
            parts=dict(base_soft=soft,control_cost=controls,collision=self.r4['collision_scale']*base['collision'],
                offroad=base['offroad'],road_rule=base['road_rule'],task_violation=-20.*task,
                risk_reward=risk,response_reward=response,severe=-2.*signal['severe']*self.dt)
            rewards[aid]=math.fsum(parts.values());details[aid]=parts;adversarial[aid]=risk+response+parts['severe']
            if not math.isfinite(rewards[aid]):raise FloatingPointError('nonfinite R4 reward')
            for k,v in parts.items():self.role_totals[aid][k]=self.role_totals[aid].get(k,0.)+v
        self.utility_total+=sum(rewards.values())/2-math.fsum(geometry.values())
        return rewards,details,adversarial

    def metrics(self,eligible):
        m=NormalDrivingCriticalReward.metrics(self,eligible)
        keys=set().union(*(v.keys() for v in self.role_totals.values()))
        m['team_reward_components']={k:sum(v.get(k,0.) for v in self.role_totals.values())/2 for k in keys}
        m['individual_reward_components']=self.role_totals
        m['individual_risk_metrics']=self.tracker.metrics()
        m['reward_semantics']='individual_three_dimension_r4'
        m['team_cost_total']=sum(max(0.,-v) for d in self.role_totals.values() for v in d.values())/2
        return m
