"""Learned acceleration/lane decisions, deterministic native lateral execution."""
import numpy as np
from gymnasium import spaces
from highway_env.vehicle.controller import ControlledVehicle
from .straight.environment import StraightEnv

KEEP,LEFT,RIGHT=0,1,2


class HybridEnv(StraightEnv):
    def __init__(self,config,render_mode=None):
        super().__init__(config,render_mode)
        self.observation_space=spaces.Dict(dict(
            state=spaces.Box(-3.,3.,shape=(config.state_dim,),dtype=np.float32),
            lane_mask=spaces.MultiBinary(3)))
        self.action_space=spaces.Dict(dict(
            acceleration=spaces.Box(-1.,1.,shape=(1,),dtype=np.float32),lane=spaces.Discrete(3)))

    def reset(self,*,seed=None,options=None):
        self._target_lane=None;self._maneuver_active=False;self._settled_time=0.
        self._lateral_controller=None;self._steering=0.
        self._lane_changes_started=0;self._cooldown_until=0.
        self._attack_ready=False;self._cut_launch=None;self._launch_serial=0
        super().reset(seed=seed,options=options)
        self._target_lane=self._lane(self.attacker)
        self._initial_relation='behind' if self.attacker.position[0]<self.target.position[0] else 'ahead'
        self._update_task_phase()
        return self._observation(),self._info()

    def lane_mask(self):
        if (self._maneuver_active or self.tracker.time<self._cooldown_until-1e-8
                or (self.config.level==1 and self._lane_changes_started>=self.config.l1_max_lane_changes)):
            return np.array([True,False,False])
        lane=self._lane(self.attacker)
        return np.array([True,lane>0,lane<1])

    def _observation(self):
        base=super()._observation()
        target=self._lane(self.attacker) if self._target_lane is None else self._target_lane
        execution=(1.+self._settled_time/self.config.lane_settle_duration) if self._maneuver_active else 0.
        budget=(self.config.l1_max_lane_changes-self._lane_changes_started)/self.config.l1_max_lane_changes if self.config.level==1 else 1.
        cooldown=max(0.,self._cooldown_until-self.tracker.time)/max(self.config.l1_lane_cooldown,self.config.l2_lane_cooldown)
        pending=self.tracker.pending
        launch=self._cut_launch
        age=0. if pending is None else min(1.,(self.tracker.time-pending['time'])/self.config.cut_response_window)
        state=np.concatenate([base,np.array([(4.*target-self.attacker.position[1])/4.,
            execution,budget,cooldown,float(self.tracker.rearmed),float(pending is not None),age,
            0. if pending is None else pending['speed']/30.,
            0. if pending is None else pending['acceleration']/10.,
            0. if pending is None else pending['ref']/30.,
            0. if pending is None else pending['brake_duration']/.2,
            min(self.tracker.exit_time/self.config.cut_exit_rearm,1.),
            float(self._attack_ready),float(launch is not None),
            0. if launch is None else min((self.tracker.time-launch['time'])/4.,1.),
            0. if launch is None else launch['speed']/30.,
            0. if launch is None else launch['acceleration']/10.,
            0. if launch is None else launch['ref']/30.,
            0. if pending is None else min((self.tracker.time-pending['baseline_time'])/4.,1.)],dtype=np.float32)])
        tracker=self.tracker
        lane=self._lane(self.attacker) if tracker.style_lane is None else tracker.style_lane
        extra=np.array([float(tracker.style_active),(4.*lane-self.attacker.position[1])/4.,
            tracker.style_active_time/self.config.duration,tracker.style_band_time/self.config.duration,
            tracker.style_lane_time/self.config.duration,tracker.front_effect_time/self.config.duration,
            tracker.front_exposure/self.config.duration,0. if tracker.style_start_time is None else tracker.style_start_time/self.config.duration],dtype=np.float32)
        state=np.concatenate([state,extra])
        return dict(state=np.clip(state,-3.,3.).astype(np.float32),lane_mask=self.lane_mask())

    def _info(self):
        info=super()._info()
        info.update(lane_target=self._target_lane,lane_busy=self._maneuver_active,
                    lane_mask=self.lane_mask().tolist(),controller_steering=self._steering,
                    lane_changes_started=self._lane_changes_started,
                    lane_changes_remaining=max(0,self.config.l1_max_lane_changes-self._lane_changes_started) if self.config.level==1 else None,
                    lane_cooldown_remaining=max(0.,self._cooldown_until-self.tracker.time),
                    dangerous_cut_count=self.tracker.dangerous_cut_count,
                    behind_positive_impact=self.tracker.behind_positive_impact,
                    blocking_time_10s=self.tracker.blocking_time_10s,
                    early_success=self.tracker.early_success,
                    attack_ready=self._attack_ready,
                    initial_relation=getattr(self,'_initial_relation',None),
                    style_active=self.tracker.style_active,style_success=self.tracker.style_success,
                    impact_success=self.tracker.impact_success,style_active_time=self.tracker.style_active_time,
                    style_speed_fraction=self.tracker.style_band_time/max(self.tracker.style_active_time,1e-8),
                    style_lane_fraction=self.tracker.style_lane_time/max(self.tracker.style_active_time,1e-8),
                    front_effect_time=self.tracker.front_effect_time,
                    causal_cut_diag=dict(self.tracker.causal_diag))
        return info

    def _update_task_phase(self):
        super()._update_task_phase()
        a,v=self.attacker,self.target;cfg=self.config
        gap=float(a.position[0]-v.position[0]-5.)
        thw=gap/max(v.speed,2.)
        adjacent=(self._lane(a)!=self._lane(v)
            and abs(a.position[1]-4*self._lane(a))<.5 and abs(a.heading)<np.deg2rad(10))
        if not self.tracker.rearmed:self._attack_ready=False
        elif self._attack_ready:
            if thw<cfg.preparation_release_low or thw>cfg.preparation_release_high:
                self._attack_ready=False
        elif adjacent and cfg.preparation_thw_low<=thw<=cfg.preparation_thw_high:
            self._attack_ready=True
        self._l2_ready=self._attack_ready
        if self._cut_launch is not None and (self.tracker.time-self._cut_launch['time']>4.
                or self.tracker.cut_count>self._cut_launch['cuts_before']):
            self._cut_launch=None

    def _task_targets(self):
        a,v=self.attacker,self.target
        gap=float(a.position[0]-v.position[0]-5.)
        ref=max(float(self.reference.vehicles[1].speed),5.)
        if self.tracker.style_active:
            return dict(goal_y=4.*self.tracker.style_lane,wanted_gap=gap,wanted_speed=self.config.slow_speed_ratio*ref)
        if self.config.level==1:
            staging=gap<self.config.slow_entry_time_gap*max(v.speed,5.)
            goal_y=4.*(1-self._lane(v)) if staging else 4.*self._lane(v)
            wanted_gap=1.8*ref
            wanted_speed=(v.speed if gap<=0 and abs(a.position[1]-v.position[1])<2.
                          else min(28.,v.speed+4.)) if staging else max(v.speed,ref)
        else:
            committing=self._cut_launch is not None and self._maneuver_active
            staging=not self._attack_ready and not committing
            goal_y=4.*(1-self._lane(v)) if staging else 4.*self._lane(v)
            wanted_gap=1.0*max(v.speed,5.)
            wanted_speed=(v.speed if gap<2. and abs(a.position[1]-v.position[1])<2.
                          else min(28.,v.speed+4.)) if gap<wanted_gap else v.speed
        return dict(goal_y=goal_y,wanted_gap=wanted_gap,wanted_speed=wanted_speed)

    def _phi(self):
        """Bounded state potential, with visible hysteretic preparation state."""
        targets=self._task_targets();a,v=self.attacker,self.target
        gap=float(a.position[0]-v.position[0]-5.)
        goal_y=targets['goal_y'];wanted_gap=targets['wanted_gap'];wanted_speed=targets['wanted_speed']
        error=(np.hypot(1.,(gap-wanted_gap)/20.)-1.
               +.6*abs(a.position[1]-goal_y)/4.+.2*abs(a.speed-wanted_speed)/10.)
        return -float(np.tanh(error/3.))

    def step(self,action):
        if self.ended:raise RuntimeError('reset required')
        if not isinstance(action,dict) or set(action)!= {'acceleration','lane'}:
            raise ValueError('Hybrid action requires acceleration and lane')
        acceleration=np.asarray(action['acceleration'],dtype=float)
        lane=action['lane']
        if (acceleration.shape!=(1,) or not np.isfinite(acceleration).all()
                or abs(acceleration[0])>1.+1e-6 or type(lane) not in (int,np.int64,np.int32)
                or lane not in (KEEP,LEFT,RIGHT)):
            raise ValueError('Invalid hybrid action')
        if not self.lane_mask()[lane]:raise ValueError('Lane action is masked in this state')
        if lane!=KEEP:
            self._lane_changes_started+=1
            self._target_lane=self._lane(self.attacker)+(-1 if lane==LEFT else 1)
            self._cut_launch=None
            if self.config.level==2 and self._attack_ready and self._target_lane==self._lane(self.target):
                self._launch_serial+=1
                self._cut_launch=dict(time=self.tracker.time,speed=float(self.target.speed),
                    acceleration=float(self.acceleration),ref=float(self.reference.vehicles[1].speed),
                    id=self._launch_serial,target_lane=self._lane(self.target),cuts_before=self.tracker.cut_count)
            self._maneuver_active=True;self._settled_time=0.
        a=self.attacker
        if self._lateral_controller is None:
            self._lateral_controller=ControlledVehicle(self.road,a.position.copy(),heading=a.heading,speed=a.speed)
        controller=self._lateral_controller
        controller.position=a.position.copy();controller.heading=a.heading;controller.speed=a.speed
        controller.road=self.road
        target=('0','1',self._target_lane)
        self._steering=float(np.clip(controller.steering_control(target),-.25,.25))
        before=self.tracker.time
        was_style=self.tracker.style_active
        _,reward,term,trunc,_=super().step(np.array([acceleration[0],self._steering/.25]))
        lane_cost=-(self.config.post_style_lane_cost if was_style else .05) if lane!=KEEP else 0.
        reward+=lane_cost;self.reward_components['lane_change']=lane_cost
        if self._maneuver_active:
            aligned=(abs(a.position[1]-4*self._target_lane)<.35 and abs(a.heading)<np.deg2rad(8))
            self._settled_time=self._settled_time+(self.tracker.time-before) if aligned else 0.
            if self._settled_time>=self.config.lane_settle_duration-1e-8:
                self._maneuver_active=False
                self._cooldown_until=self.tracker.time+(self.config.l1_lane_cooldown if self.config.level==1 else self.config.l2_lane_cooldown)
        return self._observation(),reward,term,trunc,self._info()
