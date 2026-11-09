"""One red / two blue physical vehicles on a two-to-one RoadNetwork."""
import os
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT','1')
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from highway_env.road.road import Road,RoadNetwork
from highway_env.road.lane import StraightLane,SineLane,LineType
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.kinematics import Vehicle
from ..common.contract import SceneFrame,VehicleFrame,physical_action
from ..common.objective import PairedObjective
from .contract import RED_IDS,BLUE_IDS,OBS_DIM,STATE_DIM,inputs,lane_y
from .scenarios import FAMILIES,starts
from .outcomes import BottleneckTracker

class BottleneckEnv(gym.Env):
    metadata={'render_modes':['rgb_array'],'render_fps':5}
    def __init__(self,task,gamma=.99,render_mode=None):
        self.task,self.gamma,self.render_mode=task,gamma,render_mode
        self.observation_space=spaces.Box(-10,10,(2,OBS_DIM),np.float32)
        self.action_space=spaces.Box(-1,1,(2,2),np.float32)

    def _road(self,seed):
        g=self.task.bottleneck;s,e=g['merge_start'],g['merge_end']
        net=RoadNetwork()
        for y in (0.,4.):
            net.add_lane('in','merge',StraightLane([0,y],[s,y],line_types=[LineType.STRIPED,LineType.STRIPED]))
        net.add_lane('merge','out',StraightLane([s,0],[e,0],line_types=[LineType.STRIPED,LineType.STRIPED]))
        net.add_lane('merge','out',SineLane([s,2],[e,2],2,np.pi/(e-s),np.pi/2,
                                         line_types=[LineType.STRIPED,LineType.CONTINUOUS]))
        net.add_lane('out','end',StraightLane([e,0],[1000,0],line_types=[LineType.CONTINUOUS,LineType.CONTINUOUS]))
        return Road(net,np_random=np.random.default_rng(seed))

    def _cars(self,road,ids):
        cars=[]
        for i in ids:
            x,y,speed=self.case['layout'][i]
            red=i in RED_IDS
            v=IDMVehicle(road,[x,y],speed=speed,target_speed=self.case['red_target_speed'] if red else self.task.normal_blue_target_speed,
                enable_lane_change=self.task.red_lane_change if red else self.task.normal_blue_lane_change,
                target_lane_index=('in','merge',int(y/4)),timer=0)
            v.identity=i;v.plan_route_to('end');cars.append(v)
        road.vehicles=cars

    def new_tracker(self):return BottleneckTracker(self.task,self.initial_x)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed);options=options or {}
        self.family=options.get('family') or str(self.np_random.choice(FAMILIES))
        self.case=starts(self.np_random,self.family,self.task,options.get('red_target'))
        self.initial_x={i:p[0] for i,p in enumerate(self.case['layout'])}
        self.origin=float(np.mean(list(self.initial_x.values())))
        self.road=self._road(seed);self.reference=self._road(seed);self.normal_reference=self._road(seed)
        self._cars(self.road,(0,1,2));self._cars(self.reference,RED_IDS);self._cars(self.normal_reference,(0,1,2))
        self.steps=0;self.ended=False
        self.tracker=self.new_tracker();self.reference_trackers=[self.new_tracker(),self.new_tracker()]
        self.tracker.update(self.snapshot())
        for tracker,road in zip(self.reference_trackers,[self.reference,self.normal_reference]):
            tracker.update(self.snapshot(road,time=0))
        self.objective=PairedObjective(self.task,self.gamma,red_ids=RED_IDS,blue_ids=BLUE_IDS,
            goal_distances=self.tracker.goal_distances)
        return inputs(self.snapshot(),self.task)[0],self._info()

    def snapshot(self,road=None,time=None):
        road=self.road if road is None else road;vehicles={}
        for v in road.vehicles:
            long,error=v.lane.local_coordinates(v.position)
            vehicles[v.identity]=VehicleFrame(v.identity,tuple(v.position),tuple(v.velocity),float(v.heading),
                float(error),float(v.lane.heading_at(long)),float(v.position[0]-self.initial_x[v.identity]),
                bool(v.crashed),not v.on_road,float(v.LENGTH))
        return SceneFrame(vehicles,self.steps/self.task.policy_frequency if time is None else time,(self.origin,0.))

    def _reference_progress(self):
        return {v.identity:float(v.position[0]-self.initial_x[v.identity]) for v in self.normal_reference.vehicles}

    def _info(self):
        _,state,links=inputs(self.snapshot(),self.task)
        safety=self.tracker.summary();normal=self._reference_progress()
        baseline=not self.reference_trackers[0].bad and self.reference_trackers[1].summary()['qualified']
        loss=normal[0]-safety['red_progress_m']
        return safety|{'state':state,'links':links,'family':self.family,'episode_steps':self.steps,'success':safety['qualified'],
            'baseline_valid':bool(baseline),'red_delay_m':loss,
            'quality':float(safety['qualified']) if self.task.attack_weight==0 else
                float(safety['qualified'] and baseline)*float(np.clip(loss/self.task.goal_distance,0,1)),
            'hard_brakes':safety['hard_brake_events'],
            'safe_event':bool(not safety['bad'] and safety['min_ttc_s']<self.task.ttc_threshold),
            'qualified_event':bool(safety['qualified'] and safety['min_ttc_s']<self.task.ttc_threshold)}

    def step(self,action):
        if self.ended:raise RuntimeError('reset required')
        if action is not None:
            action=np.asarray(action,np.float32)
            if action.shape!=(2,2):raise ValueError('expected two blue physical controls')
            commands=[physical_action(a) for a in action]
        dt=1/self.task.simulation_frequency
        for sub in range(self.task.simulation_frequency//self.task.policy_frequency):
            self.road.vehicles[0].act()
            for role,v in enumerate(self.road.vehicles[1:]):
                if action is None:v.act()
                else:
                    acceleration,steering=commands[role]
                    Vehicle.act(v,{'acceleration':acceleration,'steering':steering})
            self.road.step(dt)
            time=self.steps/self.task.policy_frequency+(sub+1)*dt
            for tracker,road in zip(self.reference_trackers,[self.reference,self.normal_reference]):
                road.act();road.step(dt)
                for v in road.vehicles:v.speed=float(np.clip(v.speed,0,30))
                tracker.update(self.snapshot(road,time))
            for v in self.road.vehicles:v.speed=float(np.clip(v.speed,0,30))
            self.tracker.normal_progress=self._reference_progress()
            self.tracker.update(self.snapshot(time=time))
        self.steps+=1;info=self._info()
        reward,parts=self.objective.step(self.snapshot(),info,self._reference_progress()[0],baseline_valid=info['baseline_valid'])
        self.ended=bool(info['bad'] or self.steps>=round(self.task.duration*self.task.policy_frequency))
        info['reward_components']=parts
        return inputs(self.snapshot(),self.task)[0],reward,self.ended,False,info

    def render(self):
        from PIL import Image,ImageDraw
        img=Image.new('RGB',(1200,330),'#18232d');d=ImageDraw.Draw(img)
        g=self.task.bottleneck;xmax=max(g['exit_x']+60,max(v.position[0] for v in self.road.vehicles)+40)
        scale=1140/xmax
        def pt(x,y):return (30+x*scale,205-y*20)
        xs=np.linspace(0,xmax,300)
        top=[pt(float(x),float(lane_y(x,self.task))+2) for x in xs]
        bottom=[pt(float(x),-2) for x in reversed(xs)]
        d.polygon(top+bottom,fill='#39434e')
        d.line(top,fill='white',width=2);d.line(bottom,fill='white',width=2)
        d.line([pt(0,2),pt(g['merge_start'],2)],fill='#aab3bc',width=2)
        for key,label,col in [('merge_start','taper start','#f0bf56'),('merge_end','single lane','#f0bf56'),('exit_x','blue goal','#8bd5a5')]:
            x,_=pt(g[key],0);d.line((x,55,x,260),fill=col,width=1);d.text((x+3,55),label,fill=col)
        for v in self.road.vehicles:
            x,y=pt(*v.position);length=max(12,5*scale)
            d.rectangle((x-length/2,y-10,x+length/2,y+10),fill='#f27676' if v.identity==0 else '#66bfff')
            d.text((x-4,y-7),str(v.identity),fill='black');d.text((x-20,y+13),f'{v.speed:.1f}m/s',fill='white')
        info=self._info()
        d.text((12,10),f'{self.family} | t={self.steps/self.task.policy_frequency:.1f}s | RED0 normal IDM | BLUE1/2 learners',fill='white')
        d.text((12,285),f'extra red loss={info["red_delay_m"]:.1f}m | success={info["success"]} | bad={info["bad"]} | retained flow={info["blue_min_progress_retention"]:.2f}',fill='white')
        return np.asarray(img)

    def close(self):pass
