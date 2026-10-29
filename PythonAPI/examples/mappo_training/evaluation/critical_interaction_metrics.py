"""Pure trajectory diagnostics: fixed-yaw constant-velocity OBB TTC and exposure.

These are surrogate conflict measurements, not crash probabilities or blame.
"""
from __future__ import annotations
import copy
import math

PROTOCOL_DEFAULT = dict(version='critical_interactions_v1', dt=.1, horizon_s=5.,
    ttc_thresholds_s=[1.,1.5,2.,3.], primary_ttc_s=1.5,
    min_event_duration_s=.3, merge_gap_s=.2, response_window_s=1.,
    hard_brake_mps2=3., red_ids=[0,1], blue_ids=[2,3])


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    return float(value)


def validate_protocol(p):
    if not isinstance(p,dict) or set(p)-set(PROTOCOL_DEFAULT):
        raise ValueError('unknown protocol fields')
    out=copy.deepcopy(PROTOCOL_DEFAULT)
    out.update(copy.deepcopy(p))
    if out['version'] != PROTOCOL_DEFAULT['version']:
        raise ValueError('unsupported protocol version')
    for key in ('dt','horizon_s','primary_ttc_s','min_event_duration_s','hard_brake_mps2'):
        out[key]=_number(out[key],key)
        if out[key]<=0: raise ValueError(f'{key} must be positive')
    for key in ('merge_gap_s','response_window_s'):
        out[key]=_number(out[key],key)
        if out[key]<0: raise ValueError(f'{key} cannot be negative')
    thresholds=out['ttc_thresholds_s']
    if not isinstance(thresholds,list) or not thresholds:
        raise ValueError('ttc_thresholds_s must be a nonempty list')
    thresholds=[_number(t,'threshold') for t in thresholds]
    if min(thresholds)<=0 or len(set(thresholds))!=len(thresholds):
        raise ValueError('thresholds must be distinct and positive')
    if out['primary_ttc_s'] not in thresholds or max(thresholds)>out['horizon_s']:
        raise ValueError('primary threshold must be included and thresholds cannot exceed horizon')
    out['ttc_thresholds_s']=sorted(thresholds)
    for key in ('red_ids','blue_ids'):
        ids=out[key]
        if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids):
            raise ValueError(f'{key} requires distinct nonnegative integer IDs')
    if set(out['red_ids']) & set(out['blue_ids']): raise ValueError('red/blue IDs overlap')
    return out


def _box(snapshot):
    try:
        loc=snapshot['location']; rot=snapshot['rotation']; vel=snapshot['velocity']
        box=snapshot['bounding_box']; ext=box['extent']; offset=box['location']
        theta=math.radians(_number(rot['yaw'],'actor yaw'))
        bx=_number(offset['x'],'bbox offset x'); by=_number(offset['y'],'bbox offset y')
        center=(_number(loc['x'],'x')+math.cos(theta)*bx-math.sin(theta)*by,
                _number(loc['y'],'y')+math.sin(theta)*bx+math.cos(theta)*by)
        yaw=theta+math.radians(_number(box['rotation']['yaw'],'bbox yaw'))
        axes=((math.cos(yaw),math.sin(yaw)),(-math.sin(yaw),math.cos(yaw)))
        extent=(_number(ext['x'],'bbox half length'),_number(ext['y'],'bbox half width'))
        if min(extent)<=0: raise ValueError('bbox half extents must be positive')
        velocity=(_number(vel['x'],'vx'),_number(vel['y'],'vy'))
        return center,axes,extent,velocity
    except (KeyError,TypeError) as error:
        raise ValueError('snapshot requires finite location/rotation/velocity and local bounding_box geometry') from error


def _dot(a,b): return a[0]*b[0]+a[1]*b[1]


def obb_ttc(a,b,horizon):
    """Continuous 2D separating-axis overlap interval; None = no predicted contact.

    Actor and bbox headings stay fixed. Centers translate with current velocity.
    Local bbox x/y offsets are rotated by actor yaw. No assumed vehicle sizes.
    """
    horizon=_number(horizon,'horizon')
    if horizon<=0: raise ValueError('horizon must be positive')
    ca,aa,ea,va=_box(a); cb,ab,eb,vb=_box(b)
    delta=(cb[0]-ca[0],cb[1]-ca[1]); relative=(vb[0]-va[0],vb[1]-va[1])
    entry,exit=0.,horizon
    for axis in aa+ab:
        radius=sum(e*abs(_dot(axis,direction)) for e,direction in zip(ea,aa))+sum(e*abs(_dot(axis,direction)) for e,direction in zip(eb,ab))
        pos,speed=_dot(delta,axis),_dot(relative,axis)
        if abs(speed)<1e-12:
            if abs(pos)>radius+1e-10:return None
            continue
        first,last=sorted(((-radius-pos)/speed,(radius-pos)/speed))
        entry=max(entry,first);exit=min(exit,last)
        if entry>exit+1e-10:return None
    return min(horizon,max(0.,entry)) if exit>=0 and entry<=horizon+1e-10 else None


def _mapping(value):
    if not isinstance(value,dict):raise ValueError('expected actor mapping')
    return {str(k):v for k,v in value.items()}


def _threshold_key(value):return str(float(value))


def analyze_trace(trace_rows,protocol):
    p=validate_protocol(protocol);dt=p['dt'];rows=list(trace_rows)
    out=dict(status='COMPLETE',reason=None,observed_steps=0,duration_s=0.,
        pair_coverage=dict(complete=0,expected=0),min_ttc_s=None,
        tet_s={_threshold_key(t):0. for t in p['ttc_thresholds_s']},
        tit_s={_threshold_key(t):0. for t in p['ttc_thresholds_s']},
        critical_event=False,events=[],red_response={},
        prediction_model='2D fixed-yaw constant-velocity oriented boxes; planar overlap includes contact',
        exposure_rule='post-tick samples weighted by dt; no extension after termination',
        response_claim='temporal association only; no causal or collision-responsibility attribution')
    def fail(status,reason):
        out.update(status=status,reason=str(reason),critical_event=None,min_ttc_s=None,
            tet_s={k:None for k in out['tet_s']},tit_s={k:None for k in out['tit_s']},events=[])
        return out
    try:
        if not rows:return fail('INCOMPLETE_TRACE','empty trace')
        blue_ids={str(i) for i in p['blue_ids']};red_ids={str(i) for i in p['red_ids']}
        has_blue=any(blue_ids & set(_mapping(r.get('active_mask',{}))) for r in rows)
        if not has_blue:
            out.update(status='NOT_APPLICABLE',reason='no configured blue actor slots in trace',critical_event=None,
                tet_s={k:None for k in out['tet_s']},tit_s={k:None for k in out['tit_s']})
            return out
        pair_last_step={}
        samples={};responses={r:[] for r in red_ids};previous={};initial=rows[0] if rows[0].get('step')==0 else None
        if initial:
            kin=initial.get('kinematics') or initial.get('case_geometry',{}).get('realized_initial_states',{})
            for aid,s in _mapping(kin).items():
                if aid in red_ids and isinstance(s,dict):previous[aid]=(0,s)
        steps=[r for r in rows if r.get('step')!=0]
        if not steps:return fail('INCOMPLETE_TRACE','no decision-step samples')
        if any(type(r.get('step')) is not int for r in steps) or [r.get('step') for r in steps]!=list(range(1,len(steps)+1)):
            return fail('INCOMPLETE_TRACE','decision steps must be contiguous starting at one')
        final_mask=steps[-1].get('next_active_mask')
        if not isinstance(final_mask,dict) or not final_mask or any(type(v) is not bool for v in final_mask.values()) or any(final_mask.values()):
            return fail('INCOMPLETE_TRACE','trace lacks the final all-inactive lifecycle mask')
        prior_next=None
        for row in steps:
            step=row['step'];active=_mapping(row.get('active_mask',{}));kin=_mapping(row.get('kinematics',{}))
            if any(type(v) is not bool for v in active.values()):raise ValueError('active mask values must be boolean')
            next_mask=_mapping(row.get('next_active_mask',{}))
            if set(next_mask)!=set(active) or any(type(v) is not bool for v in next_mask.values()):
                return fail('INCOMPLETE_TRACE','missing lifecycle slots')
            if prior_next is not None and active!=prior_next:
                return fail('INCOMPLETE_TRACE','lifecycle masks disagree between adjacent steps')
            if any(next_mask[k] and not active[k] for k in active):
                return fail('INCOMPLETE_TRACE','inactive actor reactivated')
            prior_next=next_mask
            live_red=sorted(r for r in red_ids if active.get(r,False));live_blue=sorted(b for b in blue_ids if active.get(b,False))
            out['observed_steps']+=1;out['duration_s']=out['observed_steps']*dt
            ttcs=[]
            for rid in live_red:
                snap=kin.get(rid)
                if not isinstance(snap,dict):return fail('INCOMPLETE_TRACE',f'missing active red {rid} at step {step}')
                prev=previous.get(rid);decel=None
                if prev and prev[0]==step-1:
                    old=prev[1]
                    try:
                        yaw=math.radians(_number(snap['rotation']['yaw'],'yaw'))
                        dv=(_number(snap['velocity']['x'],'vx')-_number(old['velocity']['x'],'previous vx'),
                            _number(snap['velocity']['y'],'vy')-_number(old['velocity']['y'],'previous vy'))
                        decel=max(0.,-_dot(dv,(math.cos(yaw),math.sin(yaw)))/dt)
                    except (KeyError,TypeError,ValueError):pass
                brake=snap.get('control',{}).get('brake')
                if brake is not None:brake=_number(brake,'actual brake')
                responses[rid].append(dict(time_s=step*dt,deceleration_mps2=decel,brake=brake))
                previous[rid]=(step,snap)
                for bid in live_blue:
                    out['pair_coverage']['expected']+=1
                    other=kin.get(bid)
                    if not isinstance(other,dict):return fail('INCOMPLETE_TRACE',f'missing active blue {bid} at step {step}')
                    try:ttc=obb_ttc(snap,other,p['horizon_s'])
                    except ValueError as error:return fail('MISSING_GEOMETRY',f'step {step}, pair {rid}/{bid}: {error}')
                    out['pair_coverage']['complete']+=1
                    pair_last_step[(rid,bid)]=step
                    if ttc is not None:
                        ttcs.append(ttc)
                        if out['min_ttc_s'] is None or ttc<out['min_ttc_s']:out['min_ttc_s']=ttc
                        if ttc<=p['primary_ttc_s']+1e-10:
                            samples.setdefault((rid,bid),[]).append((step,ttc))
            minimum=min(ttcs) if ttcs else None
            for threshold in p['ttc_thresholds_s']:
                if minimum is not None and minimum<=threshold+1e-10:
                    key=_threshold_key(threshold);out['tet_s'][key]+=dt
                    out['tit_s'][key]+=max(0.,threshold-minimum)*dt
        for (rid,bid),low in samples.items():
            groups=[]
            for item in low:
                if not groups or (item[0]-groups[-1][-1][0]-1)*dt>p['merge_gap_s']+1e-10:groups.append([])
                groups[-1].append(item)
            for group in groups:
                duration=len(group)*dt
                run=longest=0;prior_step=None
                for low_step,_ in group:
                    run=run+1 if prior_step is not None and low_step==prior_step+1 else 1
                    longest=max(longest,run);prior_step=low_step
                contact=any(value<=1e-10 for _,value in group)
                if not contact and longest*dt+1e-10<p['min_event_duration_s']:continue
                start=(group[0][0]-1)*dt;end=group[-1][0]*dt
                event=dict(red_id=int(rid),blue_id=int(bid),start_s=start,end_s=end,
                    kind='geometric_contact' if contact else 'predicted_conflict',
                    longest_continuous_low_ttc_s=longest*dt,
                    first_sample_s=group[0][0]*dt,last_sample_s=end,
                    low_ttc_duration_s=duration,span_duration_s=end-start,min_ttc_s=min(v for _,v in group),
                    censored=(pair_last_step[(rid,bid)]-group[-1][0])*dt<=p['merge_gap_s']+1e-10)
                window=[r for r in responses[rid] if start<r['time_s']<=end+p['response_window_s']+1e-10]
                decels=[r['deceleration_mps2'] for r in window if r['deceleration_mps2'] is not None]
                brakes=[r['brake'] for r in window if r['brake'] is not None]
                event['red_response']=dict(max_deceleration_mps2=max(decels) if decels else None,
                    max_brake=max(brakes) if brakes else None,response_samples=len(window),
                    response_window_censored=(responses[rid][-1]['time_s'] if responses[rid] else 0.)+1e-10<end+p['response_window_s'],
                    association='event onset through event end plus response_window_s; no causality claim')
                out['events'].append(event)
        out['events'].sort(key=lambda e:(e['start_s'],e['red_id'],e['blue_id']))
        out['critical_event']=bool(out['events'])
        for rid,records in responses.items():
            decels=[r['deceleration_mps2'] for r in records if r['deceleration_mps2'] is not None]
            brakes=[r['brake'] for r in records if r['brake'] is not None]
            out['red_response'][rid]=dict(max_deceleration_mps2=max(decels) if decels else None,
                hard_brake_duration_s=sum(v>=p['hard_brake_mps2'] for v in decels)*dt if decels else None,
                max_brake=max(brakes) if brakes else None,acceleration_samples=len(decels),
                events=[dict(blue_id=e['blue_id'],start_s=e['start_s'],**e['red_response']) for e in out['events'] if str(e['red_id'])==rid])
        return out
    except (ValueError,KeyError,TypeError) as error:
        return fail('INCOMPLETE_TRACE',error)
