"""Explicit hybrid-action protocol, independent of historical Gaussian steering."""
from dataclasses import dataclass
import math
from highwaytest.straight.config import StraightConfig

PROTOCOL='highway_hybrid_twin_v1'


@dataclass
class HybridConfig(StraightConfig):
    obs_dim:int=26
    approach_weight:float=0.
    guidance_weight:float=0.
    entropy_coef:float=.003
    lane_entropy_coef:float=.01
    acceleration_kl_multiplier:float=1.
    lane_kl_multiplier:float=1.
    lane_settle_duration:float=.4

    @property
    def state_dim(self):
        return self.obs_dim+2

    def validate(self):
        super().validate()
        if self.learner!='ppo':raise ValueError('Hybrid protocol supports PPO only')
        if self.obs_dim!=26:raise ValueError('Hybrid base observation must have 26 features')
        for key in ('lane_entropy_coef','acceleration_kl_multiplier','lane_kl_multiplier','lane_settle_duration'):
            value=getattr(self,key)
            if value<0 or not math.isfinite(value):raise ValueError(key)
        if self.lane_settle_duration<=0:raise ValueError('positive settle duration required')
        return self
