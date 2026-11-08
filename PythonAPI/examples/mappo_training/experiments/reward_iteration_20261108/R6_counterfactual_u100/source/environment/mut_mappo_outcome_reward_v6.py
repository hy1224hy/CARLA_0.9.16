"""R6 online feasibility costs; positive policy credit is settled after paired replay."""
import math
from environment.mut_mappo_critical_reward_v4 import RiskReward
from evaluation.r6_counterfactual import PROFILE,validate_settings


class CounterfactualReward(RiskReward):
    def __init__(self,settings=None,**kwargs):
        self.r6=validate_settings(settings or {})
        super().__init__(dict(progress_weight=0.,control_weight=0.,collision_scale=self.r6['hard_cost_scale'],
            risk_weight=0.,response_weight=0.,risk_budget=0.,response_budget=0.),**kwargs)

    def step(self,contact_ids=(),**kwargs):
        values,parts,adv=super().step(contact_ids=contact_ids,**kwargs)
        changes=[]
        for aid,p in parts.items():
            old=dict(p);base=kwargs['base_details'][aid]
            p['base_soft']=0.
            p['control_cost']=self.r6['comfort_weight']*sum(min(0.,base.get(k,0.)) for k in ('longitudinal_rate','steer_rate','overspeed'))
            for k in ('collision','offroad','road_rule'):p[k]=self.r6['hard_cost_scale']*base[k]
            p['task_violation']*=self.r6['hard_cost_scale']
            p['outcome_reward']=0.
            values[aid]=math.fsum(p.values())
            delta=values[aid]-math.fsum(old.values());changes.append(delta)
            for k,x in p.items():self.role_totals[aid][k]=self.role_totals[aid].get(k,0.)+x-old.get(k,0.)
        self.utility_total+=sum(changes)/2.
        return values,parts,adv

    def metrics(self,eligible):
        m=super().metrics(eligible)
        m.update(reward_semantics='r6_online_costs_pending_counterfactual_settlement',
            r6_settled=False,credit_protocol='three matched policy rollouts, single-blue replacement')
        return m
