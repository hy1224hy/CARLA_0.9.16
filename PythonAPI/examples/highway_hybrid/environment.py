"""Learned acceleration/lane decisions, deterministic native lateral execution."""
import numpy as np
from gymnasium import spaces
from highway_env.vehicle.controller import ControlledVehicle
from highwaytest.straight.environment import StraightEnv

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
        super().reset(seed=seed,options=options)
        self._target_lane=self._lane(self.attacker)
        return self._observation(),self._info()

    def lane_mask(self):
        if self._maneuver_active:return np.array([True,False,False])
        lane=self._lane(self.attacker)
        return np.array([True,lane>0,lane<1])

    def _observation(self):
        base=super()._observation()
        target=self._lane(self.attacker) if self._target_lane is None else self._target_lane
        execution=(1.+self._settled_time/self.config.lane_settle_duration) if self._maneuver_active else 0.
        state=np.concatenate([base,np.array([(4.*target-self.attacker.position[1])/4.,
                                            execution],dtype=np.float32)])
        return dict(state=np.clip(state,-3.,3.).astype(np.float32),lane_mask=self.lane_mask())

    def _info(self):
        info=super()._info()
        info.update(lane_target=self._target_lane,lane_busy=self._maneuver_active,
                    lane_mask=self.lane_mask().tolist(),controller_steering=self._steering)
        return info

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
            self._target_lane=self._lane(self.attacker)+(-1 if lane==LEFT else 1)
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
        _,reward,term,trunc,_=super().step(np.array([acceleration[0],self._steering/.25]))
        if self._maneuver_active:
            aligned=(abs(a.position[1]-4*self._target_lane)<.35 and abs(a.heading)<np.deg2rad(8))
            self._settled_time=self._settled_time+(self.tracker.time-before) if aligned else 0.
            if self._settled_time>=self.config.lane_settle_duration-1e-8:
                self._maneuver_active=False
        return self._observation(),reward,term,trunc,self._info()
