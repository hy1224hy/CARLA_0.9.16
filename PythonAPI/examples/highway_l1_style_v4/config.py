"""Explicit hybrid-action protocol, independent of historical Gaussian steering."""
from dataclasses import dataclass
import math
from .straight.config import StraightConfig

PROTOCOL='highway_l1_style_twin_v4'


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
    slow_speed_ratio:float=.85
    slow_speed_tolerance:float=.08
    slow_entry_time_gap:float=1.5
    slow_min_duration:float=5.
    slow_min_fraction:float=.7
    slow_min_lane_fraction:float=.85
    style_weight:float=2.
    impact_weight:float=.3
    impact_required_time:float=.5
    impact_required_delay:float=5.
    post_style_lane_cost:float=.25

    @property
    def state_dim(self):
        return self.obs_dim+27

    def validate(self):
        super().validate()
        if self.learner!='ppo':raise ValueError('Hybrid protocol supports PPO only')
        if self.obs_dim!=26:raise ValueError('Hybrid base observation must have 26 features')
        for key in ('lane_entropy_coef','acceleration_kl_multiplier','lane_kl_multiplier','lane_settle_duration'):
            value=getattr(self,key)
            if value<0 or not math.isfinite(value):raise ValueError(key)
        if self.lane_settle_duration<=0:raise ValueError('positive settle duration required')
        if self.level!=1:raise ValueError('L1 style v4 supports only level1')
        if type(self.l1_max_lane_changes) is not int or self.l1_max_lane_changes<1:raise ValueError('lane budget')
        for key in ('l1_lane_cooldown','l1_stable_front','l2_lane_cooldown','cut_entry_thw_max',
                    'cut_response_window','cut_new_speed_drop','cut_new_brake','cut_exit_rearm'):
            if getattr(self,key)<=0:raise ValueError(key)
        if not 0<self.l1_loss_min<self.l1_loss_saturation<1:raise ValueError('L1 loss thresholds')
        if not 0<self.l1_success_time<=self.l1_early_horizon<=self.duration:raise ValueError('L1 time thresholds')
        if not 0<self.preparation_release_low<self.preparation_thw_low<self.preparation_thw_high<self.preparation_release_high<=self.cut_entry_thw_max:
            raise ValueError('preparation hysteresis thresholds')
        if not 0<self.slow_speed_tolerance<self.slow_speed_ratio<1:raise ValueError('slow speed band')
        if self.slow_speed_ratio+self.slow_speed_tolerance>=1:raise ValueError('slow band must remain below reference speed')
        if not 0<self.slow_min_duration<=self.duration:raise ValueError('minimum style duration')
        if not 0<self.slow_min_fraction<=1 or not 0<self.slow_min_lane_fraction<=1:raise ValueError('style fractions')
        for key in ('slow_entry_time_gap','style_weight','impact_weight','impact_required_time','impact_required_delay','post_style_lane_cost'):
            if getattr(self,key)<=0:raise ValueError(key)
        return self
