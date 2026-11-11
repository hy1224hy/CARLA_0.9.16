"""Explicit hybrid-action protocol, independent of historical Gaussian steering."""
from dataclasses import dataclass
import math
from .straight.config import StraightConfig

PROTOCOL='highway_behavior_twin_v3'


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
    l1_max_lane_changes:int=2
    l1_lane_cooldown:float=3.
    l1_stable_front:float=.3
    l1_loss_min:float=.05
    l1_loss_saturation:float=.10
    l1_success_time:float=1.
    l1_early_horizon:float=10.
    preparation_thw_low:float=.7
    preparation_thw_high:float=2.
    preparation_release_low:float=.4
    preparation_release_high:float=2.5
    l2_lane_cooldown:float=1.
    cut_entry_thw_max:float=3.
    cut_response_window:float=1.
    cut_new_speed_drop:float=1.
    cut_new_brake:float=1.5
    cut_exit_rearm:float=.5
    geometry_weight:float=2.

    @property
    def state_dim(self):
        return self.obs_dim+19

    def validate(self):
        super().validate()
        if self.learner!='ppo':raise ValueError('Hybrid protocol supports PPO only')
        if self.obs_dim!=26:raise ValueError('Hybrid base observation must have 26 features')
        for key in ('lane_entropy_coef','acceleration_kl_multiplier','lane_kl_multiplier','lane_settle_duration'):
            value=getattr(self,key)
            if value<0 or not math.isfinite(value):raise ValueError(key)
        if self.lane_settle_duration<=0:raise ValueError('positive settle duration required')
        if self.level not in (1,2):raise ValueError('Behavior v2 pilot supports L1/L2')
        if type(self.l1_max_lane_changes) is not int or self.l1_max_lane_changes<1:raise ValueError('lane budget')
        for key in ('l1_lane_cooldown','l1_stable_front','l2_lane_cooldown','cut_entry_thw_max',
                    'cut_response_window','cut_new_speed_drop','cut_new_brake','cut_exit_rearm'):
            if getattr(self,key)<=0:raise ValueError(key)
        if not 0<self.l1_loss_min<self.l1_loss_saturation<1:raise ValueError('L1 loss thresholds')
        if not 0<self.l1_success_time<=self.l1_early_horizon<=self.duration:raise ValueError('L1 time thresholds')
        if not 0<self.preparation_release_low<self.preparation_thw_low<self.preparation_thw_high<self.preparation_release_high<=self.cut_entry_thw_max:
            raise ValueError('preparation hysteresis thresholds')
        return self
