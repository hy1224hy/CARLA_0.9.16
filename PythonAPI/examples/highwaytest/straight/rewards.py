"""Causal physical-substep events and grade-specific finite horizon rewards."""
import numpy as np

class RewardState:
    def __init__(self,config):
        self.config=config
        self.time=self.blocking_time=self.front_time=self.stop_time=self.brake_time=0.
        self.align_time=0.
        self.excess_time=0.
        self.attempt_registered=False
        self.geometric_cut=self.cut_in=self.done=self.success=False
        self.cut_time=self.candidate_time=self.cut_event_time=None
        self.candidate=False
        self.grade_violation=False
        self.components={}
        self.max_target_brake=0.
        self.speed_ratio_sum=0.;self.speed_ratio_n=0;self.avg_speed_ratio=0.
        self.cut_count=0;self.cut_first_thw=None
        self.stop_qualified=False
        self.contact_orientation='none'

    def observe(self,frame,dt):
        if self.done:raise RuntimeError('episode already terminated')
        cfg=self.config
        self.time+=dt
        ell=float(frame.get('ell',0.));d=float(np.clip(ell,-1,1))
        front=bool(frame.get('front',False));a=float(frame.get('acceleration',0.))
        self.front_time=self.front_time+dt if front else 0.
        self.brake_time=self.brake_time+dt if a<-3 else 0.
        self.stop_time=self.stop_time+dt if frame.get('target_speed',18)<2 else 0.
        self.excess_time=self.excess_time+dt if ell>cfg.mild_excess_threshold else 0.
        self.max_target_brake=min(self.max_target_brake,a)
        ratio=frame.get('speed_ratio')
        if ratio is not None and np.isfinite(ratio):
            self.speed_ratio_sum+=float(ratio);self.speed_ratio_n+=1
            self.avg_speed_ratio=self.speed_ratio_sum/self.speed_ratio_n
        if self.stop_time>2 and frame.get('reference_speed',18)>8:self.stop_qualified=True
        orientation=frame.get('contact_orientation')
        if orientation:self.contact_orientation=str(orientation)
        mild=front and .10<=ell<=.35 and a>=-3
        self.blocking_time+=dt*float(mild)
        # Target lane changes alone never start an attacker lane crossing.
        if frame.get('attacker_lane_changed',False) and frame.get('target_lane_stable',False):
            self.candidate=True;self.align_time=0.;self.attempt_registered=False
            self.candidate_time=None
        if not frame.get('target_lane_stable',True):
            self.candidate=False;self.align_time=0.;self.candidate_time=None
            if not self.cut_in:self.cut_time=None
        self.align_time=self.align_time+dt if self.candidate and front and frame.get('aligned',False) else 0.
        if self.align_time>=.4-1e-8 and not self.attempt_registered:
            self.attempt_registered=True
            self.geometric_cut=True;self.cut_time=self.time
            self.cut_count+=1
            thw=frame.get('thw',np.nan)
            if self.cut_first_thw is None and np.isfinite(thw):self.cut_first_thw=float(thw)
            self.candidate_time=self.time if .3<=frame.get('thw',100)<=1.2 else None
        fresh=False
        if not self.cut_in and self.candidate_time is not None and self.time-self.candidate_time<=1.0+1e-8:
            if self.brake_time>=.2-1e-8 or frame.get('reference_speed',18)-frame.get('target_speed',18)>=3:
                self.cut_in=True;self.cut_event_time=self.time;fresh=True
        # Dev calibration: hard braking alone (IDM jerk) is not a violation; it must
        # co-occur with an actual strong speed loss, otherwise mild blocking is
        # punished for the target's own transient braking.
        high=(self.cut_in or
              (self.brake_time>.4 and ell>cfg.mild_excess_threshold) or
              (self.stop_time>2 and frame.get('reference_speed',18)>8) or
              self.excess_time>=cfg.mild_excess_duration-1e-8)
        # L1 must not optimize prolonged large speed loss below the stop threshold.
        self.grade_violation=cfg.level==1 and high
        collision=bool(frame.get('collision',False))
        spawn_overlap=bool(frame.get('spawn_overlap',False))
        invalid=bool(frame.get('invalid',False) or not frame.get('baseline_valid',True) or (collision and spawn_overlap))
        horizon=self.time>=cfg.duration-1e-8
        self.done=invalid or collision or self.grade_violation or horizon
        h=float(np.clip(1-frame.get('thw',100)/1.5,0,1)) if self.geometric_cut and front else 0.
        cost=float(frame.get('control_cost',0.))
        if cfg.level==0:
            dense=float(np.clip(frame.get('attacker_speed',0)/cfg.target_speed,0,1))-.05*cost
        else:
            dense={1:2*cfg.effect_weight*d+float(mild),2:cfg.effect_weight*d+2*cfg.risk_weight*h,
                   3:.2*cfg.effect_weight*d+.5*cfg.risk_weight*h}[cfg.level]-.05*cost
        parts={'dense':dt/cfg.duration*dense,'event':0.,'success':0.,'failure':0.,'geometry':0.}
        if invalid:parts['failure']=-20.
        elif collision:
            parts['event']=20.*cfg.event_weight if cfg.level==3 else -20.
            self.success=cfg.level==3
        elif self.grade_violation:parts['failure']=-10.
        else:
            if fresh and cfg.level==2:parts['event']=2.*cfg.event_weight
            if horizon:
                self.success=(cfg.level==0 and frame.get('normal_progress',0)>=.5*cfg.target_speed*cfg.duration
                    or cfg.level==1 and self.blocking_time>=1.5-1e-8
                    or cfg.level==2 and self.cut_in and self.time-self.cut_event_time>=1.)
                if self.success:parts['success']={0:1.,1:4.,2:5.,3:0.}[cfg.level]*cfg.event_weight
        for key,value in parts.items():self.components[key]=self.components.get(key,0.)+value
        return dict(reward=float(sum(parts.values())),components=parts,done=self.done,
                    success=bool(self.success),grade_violation=self.grade_violation)
