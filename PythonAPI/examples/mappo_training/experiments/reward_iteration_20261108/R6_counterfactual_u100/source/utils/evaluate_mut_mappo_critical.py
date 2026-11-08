"""Versioned criticality evaluation of existing MutMAPPO checkpoints; no retraining.

live captures real actor bounding boxes without changing ticks/task/reward.
offline analyzes preserved evaluations and refuses to invent missing geometry.
"""
from __future__ import annotations
import argparse
import copy
import csv
from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import os
import subprocess
import sys
import threading


def connect_reusing_idle_world(host, port, town):
    """Reuse a matching idle map; retain ordinary load for a different town.

    This is an explicit evaluation operation, not a reward/configuration change.
    The caller still owns CarlaOwner and runs the normal idle-world preflight.
    """
    import carla
    from environment.carla_multi_agent_env import CARLA_RPC_TIMEOUT_SECONDS
    client=carla.Client(host,port)
    client.set_timeout(CARLA_RPC_TIMEOUT_SECONDS)
    world=client.get_world()
    actors=world.get_actors()
    if any(list(actors.filter(pattern)) for pattern in ('vehicle.*','sensor.*','walker.*','controller.*')):
        raise RuntimeError('map reuse requires an idle world without dynamic actors')
    current=world.get_map().name.replace('\\','/').rsplit('/',1)[-1]
    requested=str(town).replace('\\','/').rsplit('/',1)[-1]
    if current!=requested:
        world=client.load_world(town)
    return client,world


@contextmanager
def gpu_memory_capture(path):
    """Optional read-only GPU sampling; never contacts CARLA or advances time."""
    stop=threading.Event()
    def sample():
        with path.open('x',encoding='utf-8') as stream:
            while not stop.is_set():
                record={'at':datetime.now(timezone.utc).isoformat(),'scope':'whole GPU, not process attribution'}
                try:
                    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW} if os.name=='nt' else {}
                    result=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5,**kwargs)
                    record.update(returncode=result.returncode,
                        columns=['index','name','total_MiB','used_MiB','free_MiB','utilization_percent'],
                        rows=list(csv.reader(result.stdout.splitlines())),error=result.stderr.strip())
                except Exception as error:record['error']=str(error)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                if record.get('error'):return  # Unavailable telemetry must not interrupt evaluation.
                stop.wait(5)
    worker=threading.Thread(target=sample,daemon=True)
    worker.start()
    try:yield
    finally:stop.set();worker.join(timeout=6)


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
def geometry_capture(reuse_idle_world=False):
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
    if reuse_idle_world:
        from environment import carla_multi_agent_env as core
        original_connector=core._connect_and_load_world
        core._connect_and_load_world=connect_reusing_idle_world
    try: yield
    finally:
        runner.create_runtime=original
        if reuse_idle_world:core._connect_and_load_world=original_connector


def load_protocol(path):
    from evaluation.critical_interaction_metrics import PROTOCOL_DEFAULT,validate_protocol
    return validate_protocol(json.loads(Path(path).read_text(encoding='utf-8-sig')) if path else dict(PROTOCOL_DEFAULT))


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
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
    live.add_argument('--reuse-idle-world',action=argparse.BooleanOptionalAction,default=True,
        help='Keep a matching idle town (default); --no-reuse-idle-world reproduces the old reload behavior')
    offline=sub.add_parser('offline',help='Analyze raw evaluation directories, not summary-only files')
    offline.add_argument('--evaluations',type=Path,nargs='+',required=True)
    for p in (live,offline):
        p.add_argument('--protocol',type=Path,help='Optional complete protocol JSON; defaults are embedded and saved')
        p.add_argument('--output-dir',type=Path,required=True,help='New directory, never overwritten')
    # The local training branch has no server monitor module. Offline analysis
    # must also work without any simulator, learner or visualization imports.
    monitor_options_from_args=None
    if argv[:1]==['live']:
        try:
            from visualization.mut_mappo_monitor import add_monitor_arguments, options_from_args
        except ModuleNotFoundError as error:
            if error.name!='visualization.mut_mappo_monitor': raise
            live.add_argument('--monitor-mode',choices=('off',),default='off',
                help='This branch has no server monitor UI; local capture runs with monitoring off')
        else:
            add_monitor_arguments(live)
            monitor_options_from_args=options_from_args
    args=parser.parse_args(argv)
    protocol=load_protocol(args.protocol)
    from evaluation.critical_interaction_report import generate_report
    if args.command=='offline':
        result=generate_report(args.evaluations,args.output_dir,protocol)
    else:
        from evaluation.evaluate_mut_mappo import evaluate,preflight
        from training.mut_mappo_config import ROOT
        monitor_kwargs={}
        if monitor_options_from_args is not None:
            if 'monitor_options' in inspect.signature(evaluate).parameters:
                monitor_kwargs['monitor_options']=monitor_options_from_args(args)
            elif args.monitor_mode!='off':
                parser.error('This evaluator does not support monitor_options; use --monitor-mode off')
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
            print(json.dumps(dict(status='OFFLINE_VALIDATED',protocol=protocol,evaluations=plans,
                world_initialization='reuse_matching_idle_map_v1' if args.reuse_idle_world else 'reload_each_runtime_v1'),indent=2))
            return
        output=args.output_dir.resolve()
        output.mkdir(parents=True,exist_ok=False)
        (output/'protocol.json').write_text(json.dumps(protocol,indent=2,allow_nan=False),encoding='utf-8')
        (output/'plan.json').write_text(json.dumps(plans,indent=2,allow_nan=False),encoding='utf-8')
        operation=dict(schema='critical_evaluation_operation_v1',reuse_idle_world=args.reuse_idle_world,
            world_initialization='reuse_matching_idle_map_v1' if args.reuse_idle_world else 'reload_each_runtime_v1',
            evaluator_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            task_config_changed=False,checkpoint_changed=False,physics_ticks_added=0)
        (output/'operation.json').write_text(json.dumps(operation,indent=2),encoding='utf-8')
        (output/'evaluator_source.py').write_bytes(Path(__file__).read_bytes())
        dirs=[]
        from contextlib import nullcontext
        telemetry=gpu_memory_capture(output/'gpu_memory.jsonl') if args.reuse_idle_world else nullcontext()
        with telemetry,geometry_capture(reuse_idle_world=args.reuse_idle_world):
            for index,entry in enumerate(entries):
                directory=output/'evaluations'/f'eval_{index:03d}'
                evaluate(entry['config'],entry['checkpoint'],team='both',split=args.split,
                    conditions=('C0','C1','C2'),output_dir=directory,**monitor_kwargs)
                (directory/'critical_capture.json').write_text(json.dumps(dict(schema='actual_actor_bbox_v1',
                    physics_ticks_added=0,protocol=protocol,operation=operation)),encoding='utf-8')
                dirs.append(directory)
        result=generate_report(dirs,output/'analysis',protocol)
    count=sum(r['n_measured_pairs'] for r in result['team_rows'])
    partial=any(r.get('measurement_coverage',1)!=1 for r in result['team_rows'])
    print(json.dumps(dict(status=('PARTIAL_MEASUREMENT' if partial else 'MEASURED') if count else 'NO_MEASURED_PAIRS',measured_pairs=count,
        output=str(args.output_dir.resolve())),indent=2))
    if not count: raise SystemExit(2)


if __name__=='__main__': main()
