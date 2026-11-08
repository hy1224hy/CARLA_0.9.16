"""R6 policy-replacement effects. No simulator calls and no causal action labels."""
import math
from evaluation.r5_outcome_metrics import BurdenFilter

PROFILE='individual_counterfactual_outcome_v6'
DEFAULTS=dict(hard_cost_scale=.4,comfort_weight=.05,outcome_budget=8.,delay_fraction=.7,
    delay_scale_s=.5,delay_deadband_s=.02,burden_scale_s=.25,burden_deadband_s=.01,
    initial_position_tolerance_m=.05,initial_speed_tolerance_mps=.05,initial_yaw_tolerance_deg=.5)


def validate_settings(settings):
    if set(settings)-set(DEFAULTS)-{'gamma'}:raise ValueError('unknown R6 settings')
    c={**DEFAULTS,**settings}
    for k in DEFAULTS:
        if isinstance(c[k],bool) or not isinstance(c[k],(int,float)) or not math.isfinite(c[k]) or c[k]<0:
            raise ValueError('invalid R6 setting: '+k)
    if not 0<c['hard_cost_scale']<=1 or not 0<=c['comfort_weight']<=1 or not 0<=c['delay_fraction']<=1:
        raise ValueError('invalid R6 fraction')
    if any(c[k]<=0 for k in ('outcome_budget','delay_scale_s','burden_scale_s')):raise ValueError('R6 requires positive scales and budget')
    if not 0<float(c.get('gamma',.999))<=1:raise ValueError('invalid R6 gamma')
    return c


class RedOutcomeIntegral:
    """Area under normalized remaining progress, plus filtered response burden.

    Failure/censoring tails use last measured progress as an explicit endpoint
    penalty convention, never a claim of a simulated continuation. Burden is
    integrated only over measured active red states. Contacts remain separate.
    """
    def __init__(self,dt=.1,horizon=500):
        self.dt=dt;self.horizon=horizon;self.n=0;self.debt=self.burden=0.
        self.timeline=[]
        self.last={0:0.,1:0.};self.filter=BurdenFilter(dt,.3);self.contact=False

    def step(self,info):
        self.n+=1
        if self.n>self.horizon:raise ValueError('outcome collector exceeded fixed horizon')
        m=info['mut_metrics'];p={int(k):float(v) for k,v in m['route_progress'].items()}
        for i in (0,1):
            if i not in p or not math.isfinite(p[i]) or not 0<=p[i]<=1:raise ValueError('missing/invalid red progress')
        self.last={i:p[i] for i in (0,1)}
        self.contact|=bool(m.get('contact_events')) or bool(m.get('safety',{}).get('collision'))
        active={int(k):v for k,v in info['active_mask'].items()}
        snaps={int(k):v for k,v in info['post_tick_kinematics'].items()}
        b=self.filter.step(snaps,active,{i:p[i]==1. for i in (0,1)})
        self.debt+=sum(1.-p[i] for i in (0,1))*self.dt/2.
        self.burden+=sum(b.values())*self.dt/2.
        self.timeline.append(dict(t_s=self.n*self.dt,progress={str(i):p[i] for i in (0,1)},
            burden={str(i):b.get(i,0.) for i in (0,1)},active={str(i):bool(active.get(i)) for i in (0,1)}))

    def finish(self):
        if not self.n:raise ValueError('empty outcome rollout')
        tail=sum(1.-self.last[i] for i in (0,1))*max(0,self.horizon-self.n)*self.dt/2.
        return dict(red_progress_debt_s=self.debt+tail,red_burden_integral_s=self.burden,
            outcome_horizon_s=self.horizon*self.dt,endpoint_progress_debt_s=tail,any_contact=self.contact,
            endpoint_extension_convention='last measured remaining progress held to fixed horizon; burden tail zero; not a physical replay')


def paired_credit(actual,counterfactual,aid,settings=None):
    c=validate_settings(settings or {})
    if aid not in (2,3):raise ValueError('R6 supports blue IDs 2/3')
    for key in ('case_key','seed','team','condition','outcome_horizon_s'):
        if key not in actual or key not in counterfactual or actual[key]!=counterfactual[key]:raise ValueError('unmatched counterfactual '+key)
    out=dict(credit=0.,eligible=False,reason=None,delay_delta_s=None,burden_delta_s=None,
        attribution='matched simulator single-blue policy replacement; not action-level causality')
    for r in (actual,counterfactual):
        if r.get('valid') is not True or r.get('any_contact') or r.get('contact_events') or r.get('safety',{}).get('collision'):
            return dict(out,reason='invalid or contact-containing pair')
        if any(any(r.get('safety',{}).get('per_agent',{}).get(str(i),{}).values()) for i in (2,3)):
            return dict(out,reason='blue driving violation in matched pair')
    states=[r.get('case_geometry',{}).get('realized_initial_states',{}) for r in (actual,counterfactual)]
    if any(set(s)!={'0','1','2','3'} for s in states):return dict(out,reason='initial geometry unavailable')
    for i in ('0','1','2','3'):
        a,b=states[0][i],states[1][i]
        values=[a['location'][k]-b['location'][k] for k in ('x','y','z')]
        speed=abs(a['speed']-b['speed']);yaw=abs((a['rotation']['yaw']-b['rotation']['yaw']+180)%360-180)
        if not all(math.isfinite(x) for x in values+[speed,yaw]):raise ValueError('nonfinite initial geometry')
        if math.sqrt(sum(x*x for x in values))>c['initial_position_tolerance_m'] or speed>c['initial_speed_tolerance_mps'] or yaw>c['initial_yaw_tolerance_deg']:
            return dict(out,reason='initial states outside matching tolerance')
    for r in (actual,counterfactual):
        if any(not math.isfinite(r[k]) or r[k]<0 for k in ('red_progress_debt_s','red_burden_integral_s')):
            raise ValueError('invalid red outcome integral')
    d=actual['red_progress_debt_s']-counterfactual['red_progress_debt_s']
    b=actual['red_burden_integral_s']-counterfactual['red_burden_integral_s']
    positive=lambda x,dead,scale:min(1.,max(0.,x-dead)/scale)
    ds=positive(d,c['delay_deadband_s'],c['delay_scale_s'])
    bs=positive(b,c['burden_deadband_s'],c['burden_scale_s'])
    return dict(out,eligible=True,reason='matched',delay_delta_s=d,burden_delta_s=b,
        delay_score=ds,burden_score=bs,credit=c['outcome_budget']*(c['delay_fraction']*ds+(1-c['delay_fraction'])*bs))


def settle_credit(record,buffer,starts,credits):
    """Settle each episodic score at its own last active transition, once only."""
    if record.get('r6_settled'):raise ValueError('R6 credit already settled')
    record['individual_counterfactual_metrics']={str(i):credits[i] for i in (2,3)}
    for i in (2,3):
        credit=credits[i]['credit'];key=str(i)
        if buffer is not None:
            if len(buffer.data[i])<=starts[i]:
                if credit:raise ValueError('cannot pay credit to an agent with no actual actions')
            else:
                last=buffer.data[i][-1];last.reward+=credit;last.done=last.terminated=1.
                buffer.episode_bootstrap_states[i][-1]=None
        record['returns'][key]+=credit
        record['individual_reward_components'][key]['outcome_reward']=credit
        expected=math.fsum(record['individual_reward_components'][key].values())
        if not math.isclose(expected,record['returns'][key],abs_tol=1e-6):raise ValueError('R6 return/parts mismatch')
    parts=record['individual_reward_components'];keys=set().union(*(p.keys() for p in parts.values()))
    record['team_reward_components']={k:sum(p.get(k,0.) for p in parts.values())/2 for k in keys}
    record['team_utility_total']=sum(record['returns'][str(i)] for i in (2,3))/2
    record['r6_settled']=True
    record['reward_semantics']='individual_counterfactual_outcome_r6'
    record['return_semantics']='finite-horizon episodic marginal score at own last action; no bootstrap beyond settled game'
    record['max_abs_reward_residual']=0.
    return record


def summarize_records(records):
    """Keep rejection denominators and signed effects beside the clipped reward."""
    def mean(values):return sum(values)/len(values) if values else None
    roles={}
    for i in (2,3):
        key=str(i);rows=[r['individual_counterfactual_metrics'][key] for r in records]
        good=[r for r in rows if r['eligible']]
        reasons={r['reason'] for r in rows if not r['eligible']}
        roles[key]=dict(n=len(rows),eligible_n=len(good),
            rejection_counts={reason:sum(r['reason']==reason for r in rows) for reason in sorted(reasons)},
            credit_mean_all=mean([r['credit'] for r in rows]),
            delay_delta_mean_eligible_s=mean([r['delay_delta_s'] for r in good]),
            burden_delta_mean_eligible_s=mean([r['burden_delta_s'] for r in good]),
            positive_delay_rate_eligible=mean([r['delay_delta_s']>0 for r in good]),
            positive_burden_rate_eligible=mean([r['burden_delta_s']>0 for r in good]),
            reward_components_mean={k:mean([r['individual_reward_components'][key].get(k,0.) for r in records])
                for k in set().union(*(r['individual_reward_components'][key] for r in records))})
    return dict(n=len(records),roles=roles,any_contact_rate_all=mean([r['any_contact'] for r in records]),
        progress_debt_mean_all_s=mean([r['red_progress_debt_s'] for r in records]),
        endpoint_progress_debt_mean_all_s=mean([r['endpoint_progress_debt_s'] for r in records]),
        interpretation='training window is not independent evaluation; rejected pairs stay in denominator')
