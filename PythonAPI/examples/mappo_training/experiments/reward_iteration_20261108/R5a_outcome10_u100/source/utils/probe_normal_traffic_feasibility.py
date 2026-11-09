"""Small real-CARLA scripted diagnostic, never a learned C1/C2 comparison."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def target_controller(speed):
    from environment import frozen_lane_traffic as module
    original=module.FrozenLaneTraffic
    class TargetTraffic(original):
        def __init__(self,env,case):
            super().__init__(env,case)
            self.nominal={bid:float(speed) for bid in (2,3)}
    module.FrozenLaneTraffic=TargetTraffic
    try:yield
    finally:module.FrozenLaneTraffic=original


def run(output,config_path):
    from training.mut_mappo_config import load_config,validate_config,cases_for,portable_config,run_identity
    from training.checkpoint_protocol import atomic_write_json,sha256_file
    from training import train_mut_mappo as runner
    from utils.mut_mappo_session import CarlaOwner,now
    from utils.evaluate_mut_mappo_critical import geometry_capture,gpu_memory_capture
    from evaluation.evaluate_base_policy_c0 import _preflight_idle_world
    from evaluation.critical_interaction_metrics import analyze_trace,PROTOCOL_DEFAULT
    cfg=validate_config(load_config(config_path))
    cases=[c for c in cases_for(cfg,'gate') if
           (c.get('direction')=='northbound' and c.get('layout')=='sequential') or
           (c.get('direction')=='eastbound' and c.get('layout')=='crossing' and c.get('front_side')=='left')]
    if len(cases)!=4:raise ValueError('diagnostic must retain the registered two mirrored pairs')
    thresholds=ROOT/'configs/evaluation/base_driving_vnext_v1.json'
    pref=json.loads(thresholds.read_text(encoding='utf-8'))['thresholds']
    plan=dict(scope='SCRIPTED_FEASIBILITY_DIAGNOSTIC_NOT_TRAINING_OR_MAIN_C1_RESULT',
        cases=[c['case_key'] for c in cases],target_speeds_mps=[3.5,4.0],repeats=2,episodes=16,
        initial_states_changed=False,red_updated=False,learner_updates=0,
        controller_change='Only nominal PID target; same starts, goals, following/braking rules and frozen red',
        output=str(output),original_config=portable_config(cfg),identity=run_identity(cfg),
        diagnostic_source_sha256=sha256_file(Path(__file__)),preference_contract_sha256=sha256_file(thresholds),
        preference_screen_scope='Descriptive application of existing research preferences, not a new formal blue qualification')
    state=dict(status='STARTING',planned_episodes=16,completed_episodes=0,started_at=now())
    counts=dict(total=0,decision=0,setup=0);context={};rows=[];runtime=None
    with CarlaOwner(run_dir=output,purpose='scripted normal-speed feasibility diagnostic'):
        readiness=_preflight_idle_world(host=cfg['host'],port=cfg['port'],expected_version='0.9.16')
        output.mkdir(parents=True,exist_ok=False)
        (output/'traces').mkdir()
        plan['readiness']=readiness
        atomic_write_json(plan,output/'plan.json')
        (output/'source.py').write_bytes(Path(__file__).read_bytes())
        atomic_write_json(state,output/'status.json')
        def audit(event):
            n=int(event.get('ticks',1));kind=event['kind']
            counts['total']+=n;counts[kind]+=n
            runner.append_record(output/'tick_ledger.jsonl',dict(context,**event))
        try:
            with gpu_memory_capture(output/'gpu_memory.jsonl'),geometry_capture(reuse_idle_world=True):
                runtime=runner.create_runtime(cfg,condition='C1',audit_callback=audit)
                for target in (3.5,4.0):
                    with target_controller(target):
                        for index,case in enumerate(cases):
                            for repeat in range(2):
                                seed=case['seed']+10000*repeat
                                context.update(target_mps=target,case_key=case['case_key'],seed=seed)
                                trace=output/'traces'/f't{target}_{index}_{repeat}.jsonl'
                                row=runner.run_case(runtime,case,seed=seed,trace_path=trace)
                                if not row['valid']:raise RuntimeError('invalid diagnostic case; preserve evidence')
                                trajectory=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines() if line]
                                critical=analyze_trace(trajectory,PROTOCOL_DEFAULT)
                                if critical['status']!='COMPLETE':raise ValueError('unmeasured diagnostic geometry')
                                normal=True;measured=True
                                for aid in ('2','3'):
                                    d=row['driving_metrics'][aid]
                                    values=(d['above_preferred_speed_fraction'],d['actual_steer_rate_p95_per_second'],
                                            d['reference_lateral_p95_m'],d['actual_control_fraction'])
                                    measured &= all(v is not None for v in values)
                                    if measured:
                                        normal &= (values[0]<=pref['max_above_preferred_speed_fraction'] and
                                            values[1]<=pref['max_actual_steer_rate_p95_per_second'] and
                                            values[2]<=pref['max_reference_lateral_p95_m'] and
                                            values[3]>=pref['minimum_actual_control_fraction'])
                                row.update(evidence='scripted_feasibility_not_main_C1_or_learned_attack',
                                    diagnostic_target_mps=target,diagnostic_seed=seed,
                                    initial_state_overridden=False,critical_analysis=critical,
                                    descriptive_normal_screen=bool(normal) if measured else None)
                                rows.append(row);runner.append_record(output/'cases.jsonl',row)
                                state.update(status='RUNNING',completed_episodes=len(rows),last_target_mps=target,
                                             last_case=case['case_key'],actual_ticks=dict(counts),at=now())
                                atomic_write_json(state,output/'status.json')
                                print('DIAGNOSTIC_CASE_DONE',target,case['case_key'],seed,flush=True)
                runtime.close();runtime=None
            summaries=[]
            for target in (3.5,4.0):
                selected=[r for r in rows if r['diagnostic_target_mps']==target]
                summaries.append(dict(target_mps=target,n=len(selected),
                    red_success=sum(bool(r['red_success']) for r in selected),
                    blue_event_eligible=sum(bool(r['blue_eligible']) for r in selected),
                    raw_danger=sum(bool(r['critical_analysis']['critical_event']) for r in selected),
                    event_qualified_danger=sum(bool(r['blue_eligible'] and r['critical_analysis']['critical_event']) for r in selected),
                    descriptive_normal_measured=sum(r['descriptive_normal_screen'] is not None for r in selected),
                    descriptive_normal_danger=sum(bool(r['blue_eligible'] and r['critical_analysis']['critical_event'] and r['descriptive_normal_screen']) for r in selected)))
            state.update(status='COMPLETED',summaries=summaries,actual_ticks=dict(counts),ended_at=now())
            atomic_write_json(state,output/'summary.json');atomic_write_json(state,output/'status.json')
            return state
        except BaseException as error:
            if runtime:
                try:runtime.close()
                except Exception:pass
            state.update(status='FAILED_REVIEW_REQUIRED',error=str(error),actual_ticks=dict(counts),at=now())
            atomic_write_json(state,output/'status.json');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('experiments/adversarial_u1000_critical_reward_20261101_v4/configs/R2_B0_s42.json'))
    parser.add_argument('--output-dir',required=True,type=Path)
    args=parser.parse_args()
    output=args.output_dir if args.output_dir.is_absolute() else ROOT/args.output_dir
    print(json.dumps(run(output.resolve(),args.config),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
