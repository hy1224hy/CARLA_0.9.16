"""Frozen, explicit settings for single-attacker PPO experiments."""
from dataclasses import dataclass, asdict
import math

PROTOCOL='straight_attack_ml_v1'

@dataclass
class StraightConfig:
    level:int=1
    seed:int=42
    learner:str='ppo'
    alpha:float=.2
    tau:float=.005
    total_steps:int=200000
    rollout_steps:int=512
    epochs:int=4
    minibatch:int=128
    hidden:int=128
    lr:float=3e-4
    gamma:float=.99
    gae_lambda:float=.95
    clip:float=.2
    entropy_coef:float=.01
    beta:float=.02
    method:str='mutual'
    warmup_steps:int=20000
    ramp_steps:int=20000
    refresh_rounds:int=1
    eval_interval:int=20000
    eval_cases:int=32
    duration:float=20.
    policy_frequency:int=5
    simulation_frequency:int=15
    target_speed:float=18.
    target_time_gap:float=1.5
    threads:int=1
    geometry_weight:float=0.
    effect_weight:float=1.
    event_weight:float=1.
    risk_weight:float=1.
    mild_excess_threshold:float=.35
    mild_excess_duration:float=1.
    align_lateral:float=.5
    align_heading:float=10.
    align_duration:float=.4
    obs_dim:int=24
    cut_reward_cap:int=4
    cut_success_min:int=1
    l1_star:float=.20
    l2_effect_coef:float=.5
    lane_cross_bonus:float=0.
    lane_cross_cap:int=2
    tau_star:float=.8
    potential_lg:float=15.
    potential_ly:float=4.
    potential_lpsi:float=.5
    potential_lx:float=10.
    approach_weight:float=3.
    guidance_weight:float=0.
    guidance_mode:str='quality'

    def to_dict(self):return asdict(self)

    @classmethod
    def from_dict(cls,value):return cls(**value).validate()

    def validate(self):
        if self.level not in (0,1,2,3) or type(self.level) is not int:raise ValueError('level must be 0/1/2/3')
        if self.method not in ('mutual','independent'):raise ValueError('unknown method')
        if self.learner not in ('ppo','sac'):raise ValueError('unknown learner')
        if self.guidance_mode not in ('quality','tracking_cost'):raise ValueError('unknown guidance mode')
        if not 0<self.alpha or not 0<self.tau<=1:raise ValueError('invalid SAC hyperparameter')
        for name in ('total_steps','rollout_steps','epochs','minibatch','hidden','refresh_rounds',
                     'eval_interval','eval_cases','policy_frequency','simulation_frequency','threads'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError(name+' must be positive integer')
        for name in ('seed','warmup_steps','ramp_steps'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError(name+' must be nonnegative integer')
        if self.total_steps%2:raise ValueError('A+B budget must be even')
        for name,value in self.to_dict().items():
            if isinstance(value,(float,int)) and not math.isfinite(value):raise ValueError('nonfinite '+name)
        if min(self.duration,self.lr,self.target_speed,self.target_time_gap,self.clip)<=0:raise ValueError('invalid positive setting')
        if self.target_speed>30:raise ValueError('target speed exceeds physical bound')
        if not 0<self.gamma<=1 or not 0<=self.gae_lambda<=1:raise ValueError('invalid discount')
        if min(self.beta,self.entropy_coef,self.geometry_weight,self.effect_weight,self.event_weight,self.risk_weight,self.approach_weight,self.guidance_weight)<0:raise ValueError('negative reward/learning weight')
        if not 0<self.mild_excess_threshold<1 or self.mild_excess_duration<=0:raise ValueError('invalid mild grade boundary')
        if min(self.align_lateral,self.align_heading,self.align_duration)<=0:raise ValueError('invalid cut-in alignment thresholds')
        if self.obs_dim<1 or type(self.obs_dim) is not int:raise ValueError('obs_dim must be a positive integer')
        if min(self.cut_reward_cap,self.cut_success_min)<1:raise ValueError('invalid repeated-cut settings')
        if not 0<self.l1_star<self.mild_excess_threshold:raise ValueError('l1_star must sit inside the mild band')
        if min(self.l2_effect_coef,self.tau_star,self.potential_lg,self.potential_ly,self.potential_lpsi,self.potential_lx)<=0:raise ValueError('invalid potential/preference parameters')
        if self.lane_cross_bonus<0 or self.lane_cross_cap<0:raise ValueError('invalid lane-cross bonus settings')
        if self.simulation_frequency%self.policy_frequency:raise ValueError('physics must divide policy interval')
        n=self.duration*self.policy_frequency
        if n<1 or abs(n-round(n))>1e-6:raise ValueError('finite horizon needs integer decisions')
        return self
