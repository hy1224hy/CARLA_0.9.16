import json
import numpy as np
from ..evaluation.paired import evaluate_paired
from .environment import BottleneckEnv
from .contract import RED_IDS,BLUE_IDS,inputs
from .scenarios import cases

class Backend:
    def __init__(self,t,condition,family,seed,*,gamma=.99,script='normal'):
        self.task,self.condition=t,condition
        self.env=BottleneckEnv(t,gamma=gamma);self.env.reset(seed=seed,options={'family':family})
        self.road={'C0':self.env.reference,'C1':self.env.normal_reference,'C2':self.env.road}[condition]
        self.tracker=self.env.tracker if condition=='C2' else self.env.new_tracker()
        self.frame=self.env.snapshot(self.road,time=0)
        if condition!='C2':self.tracker.update(self.frame)
        self.total_return=0. if condition=='C2' else None
        self.initial_contract={'scenario':self.env.case,'geometry':t.bottleneck,
                               'normal_blue_target':t.normal_blue_target_speed,
                               'normal_blue_lane_change':t.normal_blue_lane_change,'red_lane_change':t.red_lane_change}
        red=self.road.vehicles[0]
        self.initial_contract['red_control_state']={'target_speed':float(red.target_speed),
            'target_lane':[red.target_lane_index[0],red.target_lane_index[1],int(red.target_lane_index[2])],
            'route':[[a,b,None if i is None else int(i)] for a,b,i in red.route],
            'timer':float(red.timer),'lane_change':bool(red.enable_lane_change)}
        if condition=='C2':
            for role,v in enumerate(self.road.vehicles[1:]):
                if script=='both' or script==('A' if role==0 else 'B'):
                    v.target_speed=16.
                elif script!='normal' and script not in ('A','B'):
                    raise ValueError('unknown scripted control')

    def policy_inputs(self):return inputs(self.frame,self.task)

    def step(self,actions=None):
        if self.condition=='C2':
            _,r,_,_,_=self.env.step(actions)
            self.total_return+=float(r.mean());self.frame=self.env.snapshot()
        else:
            dt=1/self.task.simulation_frequency
            for _ in range(self.task.simulation_frequency//self.task.policy_frequency):
                self.road.act();self.road.step(dt)
                for v in self.road.vehicles:v.speed=float(np.clip(v.speed,0,30))
                self.frame=self.env.snapshot(self.road,time=self.frame.time+dt)
                self.tracker.normal_progress={v.identity:v.progress for v in self.frame.vehicles.values() if v.identity in BLUE_IDS}
                self.tracker.update(self.frame)
        return self.frame

    def render(self):return self.env.render()
    def close(self):self.env.close()

def evaluate(model,t,split,per_family,*,gamma=.99,intervention='normal',drop_probability=0,
             capture=False,script='normal',normal_model=None):
    if normal_model is not None:
        raise ValueError('v4 flow retention is defined against fixed IDM C1; learned C1 is not supported')
    return evaluate_paired(model,t,split,per_family,gamma=gamma,intervention=intervention,
        drop_probability=drop_probability,capture=capture,
        backend_factory=lambda c,f,s:Backend(t,c,f,s,gamma=gamma,script=script),
        backend_contract={'backend':'highway','layout':'two_to_one','red_ids':list(RED_IDS),
                          'blue_ids':list(BLUE_IDS),'geometry':t.bottleneck},
        case_list=cases(split,per_family),red_ids=RED_IDS,blue_ids=BLUE_IDS)
