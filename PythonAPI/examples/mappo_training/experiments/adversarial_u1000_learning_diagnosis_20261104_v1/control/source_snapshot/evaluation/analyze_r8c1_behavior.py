"""Independent trace consistency audit and per-case behavioral analysis."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from training.action_protocol import action_tanh_to_env
from training.checkpoint_protocol import sha256_file,atomic_write_json
from evaluation.diagnose_r8c1_behavior import preflight


def analyze(output):
    config=json.loads((output/'resolved_config.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text())
    assert manifest['status']=='COMPLETED' and manifest['completed_cases']==54
    assert manifest['cleanup']['world_timing_restored']
    assert manifest['cleanup']['vehicle_actor_count']==manifest['cleanup']['sensor_actor_count']==0
    actors=preflight(config)
    records=[json.loads(l) for l in (output/'episodes.jsonl').read_text().splitlines()]
    assert [r['case'] for r in records]==config['cases']
    summaries=[]; groups=defaultdict(list); traces={}
    max_residual=max_control_error=max_policy_error=0.
    control_mismatches=[]; control_samples=0
    for r in records:
        case=r['case']; path=output/'traces'/f"{case['id']}.jsonl"
        assert sha256_file(path)==r['trace_sha256']
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        reset=rows[0]; steps=rows[1:]; traces[case['id']]=rows
        assert len(steps)==r['steps'] and reset['case']==case
        prev_obs=reset['observation']; prev_frame=reset['frame']; active={'0':True,'1':True}
        totals={'0':0.,'1':0.}; counts={'0':0,'1':0}
        assert r['actor_hashes_before']==r['actor_hashes_after']==config['checkpoints'][case['checkpoint']]['actor_hashes']
        for index,s in enumerate(steps,1):
            assert s['step']==index and s['frame']==prev_frame+1
            assert s['observation']==prev_obs
            assert set(s['policy_action'])=={a for a in active if active[a]}
            for aid,a in s['policy_action'].items():
                counts[aid]+=1
                residual=abs(s['reward'][aid]-sum(s['info']['reward_details'][aid].values()))
                max_residual=max(max_residual,residual); assert residual<1e-6
                totals[aid]+=s['reward'][aid]
                expected=action_tanh_to_env(np.asarray(a,dtype=np.float32))
                assert np.allclose(expected,s['environment_action'][aid],atol=1e-7)
                post=s['post'][aid]
                if post:
                    control_samples+=1
                    error=float(np.max(np.abs(expected-np.asarray(post['control']))))
                    max_control_error=max(max_control_error,error)
                    if error>=1e-5:
                        control_mismatches.append({'case':case['id'],'step':index,'agent':aid,
                            'command':expected.tolist(),'readback':post['control'],
                            'pre_control':s['pre'][aid]['control'] if s['pre'][aid] else None,
                            'done':s['done'][aid]})
                with torch.no_grad():
                    x=torch.tensor(s['observation'][aid],dtype=torch.float32).unsqueeze(0)
                    actor=actors[case['checkpoint']][int(aid)]
                    mean=actor.get_deterministic_action(x)[0].numpy()
                    error=float(np.max(np.abs(mean-np.asarray(s['deterministic_action'][aid]))))
                    max_policy_error=max(max_policy_error,error); assert error<1e-6
                    if case['mode']=='deterministic': assert np.allclose(a,mean,atol=1e-6)
                term=s['info']['terminated'][aid]; trunc=s['info']['truncated'][aid]
                assert not(term and trunc) and s['done'][aid]==(term or trunc)
                assert s['info']['active_mask'][aid]==(active[aid] and not s['done'][aid])
            active=s['info']['active_mask']; prev_obs=s['next_observation']; prev_frame=s['frame']
        assert not any(active.values())
        assert steps[-1]['info']['episode_end_reason']==r['reason']
        for aid in totals:
            assert abs(totals[aid]-r['agents'][aid]['return'])<1e-6
            assert counts[aid]==r['agents'][aid]['steps']
        summary={'case':case,'reason':r['reason'],'steps':r['steps'],'agents':{}}
        for aid in ('0','1'):
            ss=[s for s in steps if aid in s['policy_action']]
            dg=[s['info']['reward_diagnostics'][aid] for s in ss]
            first=ss[:50]
            summary['agents'][aid]={k:r['agents'][aid][k] for k in ('mean_speed','brake_rate','low_speed_rate','longest_low_run','first_motion_step','final_completion')}
            summary['agents'][aid].update(
                first_action=ss[0]['policy_action'][aid],reset_observation=reset['observation'][aid],
                reset_kinematics=reset['kinematics'][aid],
                max_lateral=max(d['lateral_distance_m'] for d in dg),
                first50_mean_steer=statistics.mean(s['policy_action'][aid][1] for s in first),
                first50_mean_longitudinal=statistics.mean(s['policy_action'][aid][0] for s in first),
                first_offroad_step=next((s['step'] for s in ss if s['info']['reward_diagnostics'][aid]['offroad_terminated']),None),
                first_low_speed_step=next((s['step'] for s in ss if s['info']['reward_diagnostics'][aid]['speed_mps']<0.2),None))
        summaries.append(summary)
        groups[(case['checkpoint'],case['scenario'],case['mode'])].append(summary)
    grouped=[]
    for key,rs in groups.items():
        grouped.append({'checkpoint':key[0],'scenario':key[1],'mode':key[2],
            'outcomes':dict(Counter(r['reason'] for r in rs)),'steps':[r['steps'] for r in rs],
            'a0_speed':statistics.mean(r['agents']['0']['mean_speed'] for r in rs),
            'a1_speed':statistics.mean(r['agents']['1']['mean_speed'] for r in rs),
            'a1_low_rate':statistics.mean(r['agents']['1']['low_speed_rate'] for r in rs),
            'a1_brake_rate':statistics.mean(r['agents']['1']['brake_rate'] for r in rs)})
    audit={'status':'HOLD' if control_mismatches else 'PASS','artifact_consistency':'PASS',
        'control_alignment':'HOLD' if control_mismatches else 'PASS',
        'control_samples':control_samples,'control_mismatches':control_mismatches,
        'validated_cases':54,'max_reward_residual':max_residual,
        'max_actual_control_error':max_control_error,'max_recomputed_mean_action_error':max_policy_error,
        'limits':'Trace consistency only; not simulator determinism or statistical capability GO.'}
    atomic_write_json(audit,output/'trace_validation.json')
    atomic_write_json({'groups':grouped,'cases':summaries},output/'behavior_analysis.json')
    try:
        import matplotlib
    except ModuleNotFoundError:
        print(json.dumps({'audit':audit,'groups':grouped,'plots':'SKIPPED: matplotlib unavailable'},ensure_ascii=False,indent=2))
        return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for u in ('100','185','208'):
        for j,scene in enumerate(('scene01','scene04')):
            r=next(r for r in records if r['case']==next(c for c in config['cases'] if c['checkpoint']==u and c['scenario']==scene and c['mode']=='deterministic' and c['seed']==62001))
            steps=traces[r['case']['id']][1:]
            ss=[s for s in steps if '1' in s['policy_action']]
            axes[0,j].plot([s['step']*.1 for s in ss],[s['info']['reward_diagnostics']['1']['speed_mps'] for s in ss],label=f'u{u}')
            axes[1,j].plot([s['step']*.1 for s in ss],[s['policy_action']['1'][0] for s in ss],label=f'u{u}')
            axes[0,j].set_title(f'{scene}: agent1 speed'); axes[1,j].set_title(f'{scene}: agent1 signed action')
    for ax in axes.flat: ax.legend(); ax.grid(alpha=.3); ax.set_xlabel('seconds')
    fig.tight_layout(); fig.savefig(output/'slow_agent.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7))
    for scene in ('scene01','north_length45','north_length45_horizon500','north_length45_horizon500_a1speed4','scene05'):
        c=next(c for c in config['cases'] if c['checkpoint']=='208' and c['scenario']==scene and c['mode']=='deterministic' and c['seed']==62001)
        ss=[s for s in traces[c['id']][1:] if '0' in s['policy_action']][:50]
        axes[0].plot([s['step']*.1 for s in ss],[s['policy_action']['0'][1] for s in ss],label=scene)
        axes[1].plot([s['step']*.1 for s in ss],[s['info']['reward_diagnostics']['0']['lateral_distance_m'] for s in ss],label=scene)
    axes[0].set_ylabel('agent0 steer'); axes[1].set_ylabel('lateral distance (m)')
    for ax in axes: ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_xlabel('seconds')
    fig.tight_layout(); fig.savefig(output/'north_first50.png',dpi=150); plt.close(fig)
    print(json.dumps({'audit':audit,'groups':grouped},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output'); a=parser.parse_args()
    torch.set_num_threads(1); analyze(Path(a.output))
