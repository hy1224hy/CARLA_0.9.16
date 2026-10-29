"""Versioned criticality evaluation of existing MutMAPPO checkpoints; no retraining.

live captures real actor bounding boxes without changing ticks/task/reward.
offline analyzes preserved evaluations and refuses to invent missing geometry.
"""
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path


def enrich_snapshots(env, snapshots):
    result=copy.deepcopy(snapshots)
    vehicles=getattr(env,'vehicles',[])
    for aid, state in result.items():
        index=int(aid)
        if index>=len(vehicles): continue
        vehicle=vehicles[index]
        if vehicle is None or not vehicle.is_alive: continue
        try:
            box=vehicle.bounding_box
            state['bounding_box']={
                'extent':{k:float(getattr(box.extent,k)) for k in ('x','y','z')},
                'location':{k:float(getattr(box.location,k)) for k in ('x','y','z')},
                'rotation':{k:float(getattr(box.rotation,k)) for k in ('yaw','pitch','roll')}}
        except (AttributeError,RuntimeError):
            # Missing geometry remains unmeasured, never use default vehicle sizes.
            continue
    return result


@contextmanager
def geometry_capture():
    """Scoped serial evaluator adapter; core module files/identities are untouched."""
    from training import train_mut_mappo as runner
    original=runner.create_runtime
    def instrumented(*args,**kwargs):
        runtime=original(*args,**kwargs)
        env=runtime.env
        snapshot=env._snapshot_kinematics
        env._snapshot_kinematics=lambda: enrich_snapshots(env,snapshot())
        return runtime
    runner.create_runtime=instrumented
    try: yield
    finally: runner.create_runtime=original


def load_protocol(path):
    from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT,validate_protocol
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else dict(PROTOCOL_DEFAULT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    live=sub.add_parser('live',help='Serial C0/C1/C2 evaluation with real geometry capture')
    source=live.add_mutually_exclusive_group(required=True)
    source.add_argument('--experiment',type=Path)
    source.add_argument('--config',type=Path)
    live.add_argument('--checkpoint',type=Path,help='Required with --config; experiment mode uses each run manifest')
    live.add_argument('--groups',nargs='+')
    live.add_argument('--seeds',type=int,nargs='+')
    live.add_argument('--split',choices=('gate','selection','test'),default='gate')
    live.add_argument('--validate-only',action='store_true')
    offline=sub.add_parser('offline',help='Analyze raw evaluation directories, not summary-only files')
    offline.add_argument('--evaluations',type=Path,nargs='+',required=True)
    for p in (live,offline):
        p.add_argument('--protocol',type=Path,help='Optional complete protocol JSON; defaults are embedded and saved')
        p.add_argument('--output-dir',type=Path,required=True,help='New directory, never overwritten')
    # Import monitor's parser only for live command. Does not create a runtime.
    from visualization.mut_mappo_monitor import add_monitor_arguments
    add_monitor_arguments(live)
    args=parser.parse_args()
    protocol=load_protocol(args.protocol)
    from evaluation.critical_interaction_report import generate_report
    if args.command=='offline':
        result=generate_report(args.evaluations,args.output_dir,protocol)
    else:
        from evaluation.evaluate_mut_mappo import evaluate,preflight
        from training.mut_mappo_config import ROOT
        from visualization.mut_mappo_monitor import options_from_args
        if args.config:
            if not args.checkpoint or args.groups or args.seeds:
                parser.error('--config requires --checkpoint and does not accept group/seed filters')
            entries=[dict(config=args.config,checkpoint=args.checkpoint)]
        else:
            if args.checkpoint: parser.error('--checkpoint is only used with --config')
            from utils.run_first_point_experiments import load_experiment
            _,_,runs=load_experiment(args.experiment,args.groups,args.seeds)
            entries=[]
            for run in runs:
                directory=Path(run['run_dir'])
                manifest=json.loads((directory/'checkpoint_manifest.json').read_text(encoding='utf-8-sig'))
                checkpoint=(directory/manifest['checkpoint']).resolve()
                if not checkpoint.is_relative_to(directory.resolve()): raise ValueError('checkpoint escapes run directory')
                if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=manifest['sha256']:
                    raise ValueError('checkpoint manifest SHA256 mismatch')
                entries.append(dict(config=run['config'],checkpoint=checkpoint))
        # Resolve/validate every config and model BEFORE acquiring CARLA or creating output.
        plans=[]
        for entry in entries:
            resolved=preflight(entry['config'],entry['checkpoint'],split=args.split,team='both',conditions=('C0','C1','C2'))
            if abs(float(resolved['config']['task'].get('timestep',.1))-protocol['dt'])>1e-9:
                raise ValueError('protocol dt differs from task timestep')
            plans.append(resolved['metadata'])
        if args.validate_only:
            print(json.dumps(dict(status='OFFLINE_VALIDATED',protocol=protocol,evaluations=plans),indent=2))
            return
        output=args.output_dir.resolve()
        output.mkdir(parents=True,exist_ok=False)
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
        (output/'plan.json').write_text(json.dumps(plans,indent=2,allow_nan=False),encoding='utf-8')
        dirs=[]
        with geometry_capture():
            for index,entry in enumerate(entries):
                directory=output/'evaluations'/f'eval_{index:03d}'
                evaluate(entry['config'],entry['checkpoint'],team='both',split=args.split,
                    conditions=('C0','C1','C2'),output_dir=directory,monitor_options=options_from_args(args))
                (directory/'critical_capture.json').write_text(json.dumps(dict(schema='actual_actor_bbox_v1',
                    physics_ticks_added=0,protocol=protocol)),encoding='utf-8')
                dirs.append(directory)
        result=generate_report(dirs,output/'analysis',protocol)
    count=sum(r['n_measured_pairs'] for r in result['team_rows'])
    partial=any(r.get('measurement_coverage',1)!=1 for r in result['team_rows'])
    print(json.dumps(dict(status=('PARTIAL_MEASUREMENT' if partial else 'MEASURED') if count else 'NO_MEASURED_PAIRS',measured_pairs=count,
        output=str(args.output_dir.resolve())),indent=2))
    if not count: raise SystemExit(2)


if __name__=='__main__': main()
