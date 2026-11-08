"""R4 risk proxies shared by online reward and offline evaluation.

Distances are OBB support-projected clearances, not centre distances or exact
Euclidean polygon distance. Responses are associations, never causal labels.
"""
import math
from evaluation.critical_interaction_metrics import _box, obb_ttc
from environment.mut_mappo_critical_reward import band_score, DEFAULTS


def clip(x):
    return min(1., max(0., x))


def dot(a, b):
    return sum(x*y for x,y in zip(a,b))


def pair_risk(red, blue):
    rc, ra, re, rv = _box(red)
    bc, ba, be, bv = _box(blue)
    delta = (bc[0]-rc[0], bc[1]-rc[1])
    along, lateral = dot(delta,ra[0]), dot(delta,ra[1])
    support = lambda axis,axes,ext: sum(e*abs(dot(axis,a)) for a,e in zip(axes,ext))
    gap = along-re[0]-support(ra[0],ba,be)
    side = abs(lateral)-re[1]-support(ra[1],ba,be)
    aligned = dot(ra[0],ba[0]) >= math.cos(math.radians(30))
    speed = max(0.,dot(rv,ra[0]))
    closing = max(0.,dot((rv[0]-bv[0],rv[1]-bv[1]),ra[0]))
    # Explicit engineering defaults, not validated human safety distances.
    threshold = 2. + speed*1. + closing**2/(2*3.)
    front = aligned and along >= 0 and side <= 0 and gap > 0
    parallel = aligned and abs(along) < re[0]+support(ra[0],ba,be) and side > 0
    spacing = clip((threshold-gap)/threshold) if front else 0.
    if parallel:
        side_threshold = .75 + .05*speed
        spacing = clip((side_threshold-side)/side_threshold)
    ttc = obb_ttc(red,blue,5.)
    severe = ttc is not None and ttc <= .5
    return dict(ttc=ttc, ttc_score=band_score(ttc,DEFAULTS),
        spacing_score=0. if severe else spacing,
        longitudinal_gap_m=gap if front else None,
        lateral_gap_m=side if parallel else None,
        headway_s=gap/speed if front and speed >= .5 else None,
        dynamic_gap_threshold_m=threshold, contact=ttc == 0., severe=severe)


def response_measures(previous, current, dt, previous_acceleration=None):
    _,axes,_,velocity = _box(current)
    _,_,_,old_velocity = _box(previous)
    acceleration = dot(((velocity[0]-old_velocity[0])/dt,
                        (velocity[1]-old_velocity[1])/dt),axes[0])
    deceleration = max(0.,-acceleration)
    jerk = abs(acceleration-previous_acceleration)/dt if previous_acceleration is not None else None
    a,b = previous.get('control',{}),current.get('control',{})
    steer_rate = abs(b['steer']-a['steer'])/dt if 'steer' in a and 'steer' in b else None
    # Bounded, interpretable finite-difference proxy. Raw measures are retained.
    score = .5*clip(deceleration/3.) + .25*clip((jerk or 0.)/10.) + .25*clip((steer_rate or 0.)/1.)
    return dict(acceleration_mps2=acceleration,deceleration_mps2=deceleration,
        jerk_mps3=jerk,steer_rate_s=steer_rate,response_score=score)


class RiskTracker:
    def __init__(self,dt=.1):
        self.dt=dt; self.previous={}; self.previous_acc={}; self.step_no=0
        self.blocked=set(); self.last_risk={}; self.runs={}
        self.totals={i:dict(observed_s=0.,ttc_duration_s=0.,ttc_longest_s=0.,
            spacing_duration_s=0.,spacing_longest_s=0.,severe_duration_s=0.,
            risk_integral_s=0.,response_excess_integral_s=0.,response_samples=0,
            response_reference_samples=0,hard_brake_duration_s=0.,max_red_deceleration_mps2=0.,
            min_longitudinal_gap_m=None,min_headway_s=None) for i in (2,3)}

    def step(self,step,snapshots,active,safety,reference=None,contact_ids=()):
        if step != self.step_no+1:
            raise ValueError('R4 requires consecutive decision steps')
        self.step_no=step
        for aid,on in active.items():
            if on:
                if aid not in snapshots: raise ValueError('missing active R4 geometry')
                _box(snapshots[aid])
        responses={}
        for rid in (0,1):
            if active.get(rid) and rid in self.previous:
                responses[rid]=response_measures(self.previous[rid],snapshots[rid],self.dt,self.previous_acc.get(rid))
                self.previous_acc[rid]=responses[rid]['acceleration_mps2']
        result={}
        for bid in (2,3):
            if not active.get(bid): continue
            total=self.totals[bid]
            pairs={rid:pair_risk(snapshots[rid],snapshots[bid]) for rid in (0,1) if active.get(rid)}
            if bid in contact_ids or any(r in contact_ids for r in pairs) or any(p['contact'] for p in pairs.values()) or any(safety.get(bid,{}).values()):
                self.blocked.add(bid)
            allowed=bid not in self.blocked
            risk=excess=0.; ttcon=spaceon=severe=False
            associated_responses=[]; reference_seen=False
            for rid,p in pairs.items():
                good=allowed and not snapshots[rid].get('reached',False)
                tt = good and p['ttc'] is not None and .5 < p['ttc'] <= 1.5
                sp = good and p['spacing_score'] > 0
                severe |= p['severe']
                for kind,on in [('ttc',tt),('spacing',sp)]:
                    key=(bid,rid,kind);self.runs[key]=self.runs.get(key,0)+1 if on else 0
                    total[kind+'_longest_s']=max(total[kind+'_longest_s'],self.runs[key]*self.dt)
                ttcon |= tt;spaceon |= sp
                q=max(p['ttc_score'],p['spacing_score']) if good and not p['severe'] else 0.
                risk=max(risk,q)
                if q>0:self.last_risk[(bid,rid)]=step
                for key,value in [('min_longitudinal_gap_m',p['longitudinal_gap_m']),('min_headway_s',p['headway_s'])]:
                    if allowed and value is not None:
                        total[key]=value if total[key] is None else min(total[key],value)
                response=responses.get(rid)
                baseline=(reference or {}).get(str(step),{}).get(str(rid))
                associated=good and (step-self.last_risk.get((bid,rid),-10000))*self.dt<=1.
                if associated and response:
                    associated_responses.append(response)
                    total['max_red_deceleration_mps2']=max(total['max_red_deceleration_mps2'],response['deceleration_mps2'])
                    if baseline is not None:
                        reference_seen=True
                        excess=max(excess,max(0.,response['response_score']-baseline['response_score']))
                # Missing baseline is unavailable, never assumed zero.
            if severe:
                risk=excess=0.
            total['observed_s']+=self.dt
            total['ttc_duration_s']+=float(ttcon)*self.dt
            total['spacing_duration_s']+=float(spaceon)*self.dt
            total['severe_duration_s']+=float(severe)*self.dt
            total['risk_integral_s']+=risk*self.dt
            total['response_excess_integral_s']+=excess*self.dt
            if associated_responses:
                total['response_samples']+=1
                total['response_reference_samples']+=int(reference_seen)
                total['hard_brake_duration_s']+=float(any(r['deceleration_mps2']>=3. for r in associated_responses))*self.dt
            result[bid]=dict(risk=risk,response_excess=excess,severe=float(severe))
        for rid in (0,1):
            if active.get(rid): self.previous[rid]=snapshots[rid]
            else:self.previous.pop(rid,None);self.previous_acc.pop(rid,None)
        return result

    def metrics(self):
        return {str(i):dict(v,credit_blocked=i in self.blocked,
            metric_protocol='r4_risk_v1',response_claim='temporal association, not causality') for i,v in self.totals.items()}
