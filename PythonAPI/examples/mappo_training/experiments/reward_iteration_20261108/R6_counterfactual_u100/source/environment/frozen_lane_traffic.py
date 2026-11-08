"""Versioned ordinary C1 traffic, using CARLA's bundled PID controller.

No training, tick, reset, autopilot, or hidden collision shield for C2 actors.
"""
import math
from pathlib import Path
import sys


def following_target(nominal, speed, gap=None, leader_speed=0.):
    if gap is None: return nominal
    desired_gap=2. + speed
    return max(0.,min(nominal,leader_speed+.5*(gap-desired_gap)))


class FrozenLaneTraffic:
    def __init__(self, env, case):
        source=Path(__file__).resolve().parents[3]/'carla'
        bundled=Path(__file__).resolve().parents[1]/'vendor'
        if (bundled/'agents').exists():source=bundled
        if str(source) not in sys.path: sys.path.insert(0,str(source))
        from agents.navigation.controller import VehiclePIDController
        self.env=env
        self.nominal={bid:float(case['blue_spawn_offsets'][f'agent{bid}']['initial_speed_mps']) for bid in (2,3)}
        self.starts={bid:env.road_reference_waypoints[bid] for bid in (2,3)}
        self.controllers={bid:VehiclePIDController(env.vehicles[bid],
            args_lateral=dict(K_P=1.,K_I=0.,K_D=.05,dt=env.timestep),
            args_longitudinal=dict(K_P=.3,K_I=.05,K_D=0.,dt=env.timestep),
            max_throttle=.65,max_brake=.75,max_steering=.6) for bid in (2,3)}

    def controls(self, active):
        import numpy as np
        result={}
        for bid,controller in self.controllers.items():
            if not active.get(bid,False): continue
            vehicle=self.env.vehicles[bid]
            transform=vehicle.get_transform(); velocity=vehicle.get_velocity()
            start=self.starts[bid]; yaw=math.radians(start.transform.rotation.yaw)
            forward=(math.cos(yaw),math.sin(yaw)); right=(-forward[1],forward[0])
            location=transform.location; origin=start.transform.location
            distance=max(0.,(location.x-origin.x)*forward[0]+(location.y-origin.y)*forward[1])
            options=[wp for wp in start.next(distance+5.) if wp.road_id==start.road_id and wp.lane_id==start.lane_id]
            if len(options)!=1: raise RuntimeError('C1 lane-follow route is not unique')
            speed=max(0.,velocity.x*forward[0]+velocity.y*forward[1])
            leaders=[]
            for oid,other in enumerate(self.env.vehicles):
                if oid==bid or other is None or not other.is_alive: continue
                own_center=transform.transform(vehicle.bounding_box.location)
                other_transform=other.get_transform()
                p=other_transform.transform(other.bounding_box.location)
                dx,dy=p.x-own_center.x,p.y-own_center.y
                longitudinal=dx*forward[0]+dy*forward[1]
                lateral=abs(dx*right[0]+dy*right[1])
                if longitudinal>0 and lateral<1.8:
                    # Conservative projection of real boxes; no assumed sizes.
                    extents=[]
                    for actor in (vehicle,other):
                        angle=math.radians(actor.get_transform().rotation.yaw+actor.bounding_box.rotation.yaw)-yaw
                        box=actor.bounding_box
                        extents.append(abs(math.cos(angle))*box.extent.x+abs(math.sin(angle))*box.extent.y)
                    gap=longitudinal-sum(extents)
                    v=other.get_velocity()
                    leaders.append((gap,max(0.,v.x*forward[0]+v.y*forward[1])))
            gap,leader_speed=min(leaders) if leaders else (None,0.)
            target=following_target(self.nominal[bid],speed,gap,leader_speed)
            control=controller.run_step(target*3.6,options[0])
            if gap is not None and gap<=1.5:
                control.throttle=0.;control.brake=1.
            result[bid]=np.array([control.throttle,control.steer,control.brake],dtype=np.float32)
        return result
