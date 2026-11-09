"""Explicit three-car roles, SI actions, local visibility and public road geometry."""
from dataclasses import replace
import math
import numpy as np
from ..common.contract import rotation

PROTOCOL='bottleneck_paired_attack_v4'
RED_IDS=(0,)
BLUE_IDS=(1,2)
OBS_DIM,STATE_DIM=25,25
GEOMETRY={'merge_start':120.,'merge_end':220.,'exit_x':300.,
          'min_progress_retention':.85,'max_stopped_s':3.}

def task(**kwargs):
    from ..config import Task
    return Task(**({'research_protocol':PROTOCOL,'duration':25.,'goal_distance':180.,
                   'red_lane_change':True,'selection_min_qualified':.875,'bottleneck':GEOMETRY.copy()}|kwargs))

def validate_geometry(t):
    g=t.bottleneck
    if not isinstance(g,dict) or set(g)!=set(GEOMETRY):
        raise ValueError('v4 needs explicit bottleneck geometry/traffic qualification')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) for v in g.values()):
        raise ValueError('nonfinite bottleneck configuration')
    if not 80<g['merge_start']<g['merge_end']<g['exit_x']<900 or g['merge_end']-g['merge_start']<40:
        raise ValueError('invalid upstream/taper/downstream geometry')
    if not 0<g['min_progress_retention']<=1 or g['max_stopped_s']<0:
        raise ValueError('invalid traffic preservation limits')

def lane_y(x,t):
    g=t.bottleneck
    f=np.clip((np.asarray(x)-g['merge_start'])/(g['merge_end']-g['merge_start']),0,1)
    return 2*(1+np.cos(np.pi*f))

def future_heading(x,t):
    g=t.bottleneck
    if not g['merge_start']<x<g['merge_end']:return 0.
    phase=np.pi*(x-g['merge_start'])/(g['merge_end']-g['merge_start'])
    return math.atan(-2*np.pi/(g['merge_end']-g['merge_start'])*math.sin(phase))

def inputs(scene,t):
    if set(scene.vehicles)!={0,1,2}:
        raise ValueError('v4 policy requires exactly red0 and blue1/2')
    g=t.bottleneck;rows=[]
    for role,i in enumerate(BLUE_IDS):
        ego=scene.vehicles[i];x,y=ego.position
        angle=ego.heading-ego.route_heading
        row=[ego.speed/30,ego.lane_error/2,math.sin(angle),math.cos(angle),
             ego.progress/t.goal_distance,role,scene.time/t.duration]
        for j in [0,BLUE_IDS[1-role]]:
            other=scene.vehicles[j];dp=np.subtract(other.position,ego.position)
            dv=np.subtract(other.velocity,ego.velocity)
            if np.linalg.norm(dp)<=t.sensing_range:
                p,v=rotation(ego.heading)@dp,rotation(ego.heading)@dv
                row.extend([1,p[0]/50,p[1]/8,v[0]/30,v[1]/10,float(j in BLUE_IDS)])
            else:row.extend([0]*6)
        ending=bool(x<g['merge_end'] and abs(y-float(lane_y(x,t)))<abs(y))
        future=future_heading(x+max(8.,ego.speed),t) if ending else 0.
        row.extend([(g['merge_start']-x)/200,(g['merge_end']-x)/200,float(ending),
                    math.sin(future),math.cos(future),(g['exit_x']-x)/300])
        rows.append(row)
    state=[]
    for i in (0,1,2):
        v=scene.vehicles[i];xy=rotation(scene.route_heading)@np.subtract(v.position,scene.origin)
        a=v.heading-v.route_heading
        state.extend([xy[0]/t.goal_distance,xy[1]/4,v.speed/30,math.sin(a),math.cos(a),
                      float(i in BLUE_IDS),v.progress/t.goal_distance])
    state.extend([scene.time/t.duration,g['merge_start']/300,g['merge_end']/300,g['exit_x']/300])
    link=float(np.linalg.norm(np.subtract(scene.vehicles[1].position,scene.vehicles[2].position))<=t.communication_range)
    obs=np.clip(np.asarray(rows,np.float32),-10,10);state=np.asarray(state,np.float32)
    if obs.shape!=(2,OBS_DIM) or state.shape!=(STATE_DIM,) or not np.isfinite(obs).all() or not np.isfinite(state).all():
        raise ValueError('invalid three-car policy features')
    return obs,state,np.full((2,1),link,np.float32)
