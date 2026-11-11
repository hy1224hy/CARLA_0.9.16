"""Behavior v2: stable mild obstruction and entry-anchored new cut responses."""
import numpy as np


def band(x, lo, hi, half):
    """Trapezoid: 1 inside [lo,hi], linear to 0 at lo-half and hi+half."""
    if x <= lo - half or x >= hi + half:
        return 0.
    if lo <= x <= hi:
        return 1.
    if x < lo:
        return (x - (lo - half)) / half
    return (hi + half - x) / half


def tri(x, star, vmax):
    """Triangular preference: 0 at 0, 1 at star, back to 0 at vmax."""
    if x <= 0 or x >= vmax:
        return 0.
    if x <= star:
        return x / star
    return (vmax - x) / (vmax - star)


def tracking_score(error,scale):
    """Negative smooth absolute error: zero at the goal, no far-error plateau."""
    return 1.-float(np.hypot(1.,error/scale))


class LegacyRewardState:
    def __init__(self,config):
        self.config=config
        self.time=self.blocking_time=self.front_time=self.stop_time=self.brake_time=0.
        self.align_time=0.
        self.excess_time=0.
        self.attempt_registered=False
        self.geometric_cut=self.cut_in=self.done=self.success=False
        self.cut_time=self.candidate_time=None
        self.candidate=False
        self.react_checked=True
        self.grade_violation=False
        self.components={}
        self.max_target_brake=0.
        self.speed_ratio_sum=0.;self.speed_ratio_n=0;self.avg_speed_ratio=0.
        self.cut_count=0;self.cut_first_thw=None;self.last_cut_time=None
        self.cut_times=[]
        self.preparation_paid=False
        self.lane_cross_count=0
        self.stop_qualified=False
        self.contact_orientation='none'
        self.diag={'candidates':0,'geometric':0,'thw_ok':0,'thw_miss':0,'react_hit':0,'react_miss':0}



class RewardState(LegacyRewardState):
    """Entry-anchored incremental response; no direct reward for lane crossings."""
    def __init__(self,config):
        super().__init__(config)
        self.pending=None;self.rearmed=True;self.exit_time=0.
        self.dangerous_cut_count=0;self.behind_positive_impact=0.
        self.precut_front_time=0.
        self.causal_diag={'entry':0,'behind_rejected':0,'distance_rejected':0,
                          'response_hit':0,'response_miss':0,'target_changed':0,'evasive_hit':0}
        self.previous_target_speed=None;self.previous_target_acceleration=0.
        self.previous_reference_speed=None
        self.blocking_time_10s=0.;self.early_success=False;self.early_checked=False
        self.last_launch_id=None

    def observe(self,frame,dt):
        if self.done:raise RuntimeError('episode already terminated')
        cfg=self.config;self.time+=dt
        front=bool(frame.get('front',False));aligned=bool(frame.get('aligned',False))
        ell=float(frame.get('ell',0.));a=float(frame.get('acceleration',0.))
        speed=float(frame.get('target_speed',18.));ref=float(frame.get('reference_speed',18.))
        stable=front and aligned
        self.front_time=self.front_time+dt if stable else 0.
        self.brake_time=self.brake_time+dt if a<-3 else 0.
        self.stop_time=self.stop_time+dt if speed<2 else 0.
        self.excess_time=self.excess_time+dt if ell>cfg.mild_excess_threshold else 0.
        self.max_target_brake=min(self.max_target_brake,a)
        self.stop_qualified=self.stop_qualified or (self.stop_time>2 and ref>8)
        self.contact_orientation=frame.get('contact_orientation') or self.contact_orientation
        self.speed_ratio_sum+=float(frame.get('speed_ratio',1.));self.speed_ratio_n+=1
        self.avg_speed_ratio=self.speed_ratio_sum/self.speed_ratio_n
        # Re-arm only after physically separating into the adjacent lane.
        separated=frame.get('lateral',0.)>3.2 and not front
        self.exit_time=self.exit_time+dt if separated else 0.
        if self.exit_time>=cfg.cut_exit_rearm and self.pending is None:self.rearmed=True
        crossed=bool(frame.get('attacker_lane_changed',False) and frame.get('target_lane_stable',False))
        if crossed and self.rearmed:
            self.diag['candidates']+=1
            if not front:self.causal_diag['behind_rejected']+=1
            elif cfg.level==2 and (frame.get('launch_baseline') is None
                    or frame['launch_baseline']['id']==self.last_launch_id):
                self.causal_diag.setdefault('unprepared_rejected',0)
                self.causal_diag['unprepared_rejected']+=1
            elif not .3<=frame.get('thw',100.)<=cfg.cut_entry_thw_max:
                self.causal_diag['distance_rejected']+=1;self.diag['thw_miss']+=1
            else:
                launch=frame.get('launch_baseline')
                if launch is not None:self.last_launch_id=launch['id']
                self.pending=dict(time=self.time,
                    baseline_time=self.time-dt if launch is None else launch['time'],
                    speed=(speed if self.previous_target_speed is None else self.previous_target_speed) if launch is None else launch['speed'],
                    acceleration=(a if self.previous_target_speed is None else self.previous_target_acceleration) if launch is None else launch['acceleration'],
                    ref=(ref if self.previous_reference_speed is None else self.previous_reference_speed) if launch is None else launch['ref'],
                    thw=float(frame['thw']),brake_duration=0.)
                self.rearmed=False;self.causal_diag['entry']+=1;self.diag['thw_ok']+=1
                self.geometric_cut=True;self.diag['geometric']+=1
                if self.cut_first_thw is None:self.cut_first_thw=float(frame['thw'])
        fresh=False
        if self.pending is not None:
            p=self.pending
            if self.time-p['time']>cfg.cut_response_window+1e-8:
                self.pending=None;self.causal_diag['response_miss']+=1;self.diag['react_miss']+=1
            elif not frame.get('target_lane_stable',True):
                if frame.get('reference_lane_stable',True):
                    self.cut_count+=1;self.last_cut_time=self.time;self.cut_times.append(self.time)
                    self.cut_in=True;fresh=True
                    self.dangerous_cut_count+=int(.3<=p['thw']<=1.2)
                    self.causal_diag['response_hit']+=1;self.causal_diag['evasive_hit']+=1
                    self.diag['react_hit']+=1
                else:self.causal_diag['target_changed']+=1
                self.pending=None
            else:
                new_drop=(p['speed']-speed)-(p['ref']-ref)
                # Exclude continuation of entry-time braking; require a stronger new brake.
                stronger=a<min(-1.,p['acceleration']-cfg.cut_new_brake)
                p['brake_duration']=p['brake_duration']+dt if stronger else 0.
                braking_drop=max(0.,-p['acceleration'])*(self.time-p['baseline_time'])
                incremental_drop=new_drop-braking_drop
                if (front and aligned and (incremental_drop>=cfg.cut_new_speed_drop
                                           or p['brake_duration']>=.2-1e-8)):
                    self.cut_count+=1;self.last_cut_time=self.time;self.cut_times.append(self.time)
                    self.cut_in=True;fresh=True;self.pending=None
                    self.dangerous_cut_count+=int(.3<=p['thw']<=1.2)
                    self.causal_diag['response_hit']+=1;self.diag['react_hit']+=1
        self.previous_target_speed=speed;self.previous_target_acceleration=a;self.previous_reference_speed=ref
        mild=stable and self.front_time>=cfg.l1_stable_front and ell>=cfg.l1_loss_min and a>=-3 and speed>=2
        self.blocking_time+=dt*float(mild)
        if self.time<=cfg.l1_early_horizon+1e-8:self.blocking_time_10s+=dt*float(mild)
        self.grade_violation=bool(cfg.level==1 and (self.dangerous_cut_count>0
            or self.stop_time>2))
        invalid=bool(frame.get('invalid',False) or not frame.get('baseline_valid',True)
                     or (frame.get('collision',False) and frame.get('spawn_overlap',False)))
        collision=bool(frame.get('collision',False));horizon=self.time>=cfg.duration-1e-8
        if not self.early_checked and self.time>=cfg.l1_early_horizon-1e-8:
            self.early_checked=True
            self.early_success=bool(cfg.level==1 and self.blocking_time_10s>=cfg.l1_success_time-1e-8 and not invalid and not collision and not self.grade_violation)
        self.done=invalid or collision or self.grade_violation or horizon
        cost=.05*float(frame.get('control_cost',0.))
        impact=0.
        if cfg.level==1:
            if stable and self.front_time>=cfg.l1_stable_front and a>=-3 and speed>=2:
                impact=2.*cfg.effect_weight*float(np.clip(ell/cfg.l1_loss_saturation,0,1))
        else:
            self.precut_front_time=self.precut_front_time+dt if front else 0.
            if front and not self.cut_count and self.precut_front_time<=2.:
                impact=.5*cfg.risk_weight*band(frame.get('thw',100.),.5,2.5,.5)*frame.get('lane_alignment',0.)
            cost+=.5*float(frame.get('attacker_speed',18.)<3.)
        if not front:self.behind_positive_impact+=dt*max(impact,0.)
        parts=dict(dense=dt/cfg.duration*(impact-cost),approach=0.,guidance=0.,milestone=0.,
                   event=0.,success=0.,failure=0.,geometry=0.)
        if invalid:parts['failure']=-20.
        elif collision:parts['failure']=-20.
        elif self.grade_violation:parts['failure']=-10.
        else:
            if fresh and cfg.level==2 and self.cut_count<=cfg.cut_reward_cap:
                parts['event']=4.*cfg.event_weight
            if horizon:
                self.success=bool(self.blocking_time>=cfg.l1_success_time-1e-8 if cfg.level==1 else
                    sum(self.time-t>=1.-1e-8 for t in self.cut_times)>=cfg.cut_success_min)
                if self.success:parts['success']=(4. if cfg.level==1 else 5.)*cfg.event_weight
        for key,value in parts.items():self.components[key]=self.components.get(key,0.)+value
        return dict(reward=float(sum(parts.values())),components=parts,done=self.done,
                    success=self.success,grade_violation=self.grade_violation)
