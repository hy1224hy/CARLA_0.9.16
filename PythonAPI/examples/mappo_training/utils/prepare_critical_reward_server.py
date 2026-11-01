"""Portable offline preparation for the latest R2 blue reward server pilot.

No local experiment predecessor, authorization marker, CARLA connection or training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/'configs/mut_mappo/critical_reward_v2_server.json'


def resolve(path):
    path=Path(path).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def prepare(checkpoint, output_dir, *, source_config=None, expected_sha256=None,
            device='cuda', seed=42, train_ticks=24000, host='127.0.0.1', port=2000,
            case_manifest=None, reference_dir=None, c1_controller=None):
    from training.checkpoint_protocol import load_checkpoint_file, atomic_write_json, sha256_file
    from training.mut_mappo_config import load_config, portable_config, validate_config
    from environment.mut_mappo_critical_reward_v2 import PROFILE
    output=resolve(output_dir)
    if output.exists():raise FileExistsError('output exists; choose a new experiment directory')
    if type(train_ticks) is not int or train_ticks<=0 or type(seed) is not int:
        raise ValueError('train ticks must be positive and seed must be an integer')
    if device not in ('cpu','cuda') or not host or type(port) is not int or not 1<=port<=65535:
        raise ValueError('invalid device or RPC endpoint')
    checkpoint=resolve(checkpoint)
    payload,digest=load_checkpoint_file(checkpoint,map_location='cpu')
    if expected_sha256 is not None:
        expected=expected_sha256.strip().lower()
        if len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected) or expected!=digest:
            raise ValueError('checkpoint does not match expected SHA256')
    source=resolve(source_config) if source_config is not None else DEFAULT_CONFIG
    config=portable_config(load_config(source))
    config.pop('diagnostic_authorization',None)
    if config.get('task',{}).get('reward_profile')!=PROFILE:
        raise ValueError('source configuration must use the R2 normal-driving critical reward')
    config.update(device=device,seed=seed,host=host,port=port,research_status='candidate')
    config['red_checkpoint']={'path':str(checkpoint),'sha256':digest,
        'capability_status':'weak_red_engineering_candidate_not_formally_qualified'}
    model=payload.get('model_spec',{})
    training=payload.get('training_config',{})
    if 'hidden_dim' not in model or 'gamma' not in training:
        raise ValueError('source checkpoint requires registered model_spec and training_config')
    config['learning'].update(hidden_dim=int(model['hidden_dim']),gamma=float(training['gamma']))
    for key in ('min_log_std','max_log_std','init_log_std'):
        if key in training:config['learning'][key]=training[key]
    config['task']['team_reward']['gamma']=config['learning']['gamma']
    config['task']['team_reward']['reference_dir']=str(resolve(reference_dir) if reference_dir is not None else output/'references')
    if case_manifest is not None:config['case_manifest']=str(resolve(case_manifest))
    if c1_controller is not None:config['task']['c1_controller']=c1_controller
    config['run']['train_tick_budget']=train_ticks
    checked=validate_config(config)
    # All checks above are read-only; never create output for a rejected upload/config.
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(config,output/'config.json')
    record=dict(status='PREPARED_NOT_TRAINED',schema='critical_reward_server_preparation_v1',
        profile=PROFILE,config=str(output/'config.json'),run_dir=str(output/'run'),
        source_config=str(source),checkpoint=str(checkpoint),red_sha256=digest,
        source_update=payload.get('update'),device=device,seed=seed,train_tick_budget=train_ticks,
        reference_dir=config['task']['team_reward']['reference_dir'],
        case_manifest=config['case_manifest'],c1_controller=config['task'].get('c1_controller','frozen_actor'),
        config_sha256=sha256_file(output/'config.json'),validation=checked['_root_validation'],
        scope='weak_red_engineering_reward_pilot_not_formal_driving_qualification',
        initialization='fresh blue learners from supplied base actors; not resume of an old reward run',
        carla_connected=False,training_launched=False)
    atomic_write_json(record,output/'preparation.json')
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--output-dir',required=True,type=Path)
    parser.add_argument('--expected-sha256')
    parser.add_argument('--source-config',type=Path)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--train-ticks',type=int,default=24000)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--case-manifest',type=Path)
    parser.add_argument('--reference-dir',type=Path)
    parser.add_argument('--c1-controller',choices=('frozen_actor','lane_pid_v1'))
    args=parser.parse_args()
    result=prepare(args.checkpoint,args.output_dir,source_config=args.source_config,
        expected_sha256=args.expected_sha256,device=args.device,seed=args.seed,
        train_ticks=args.train_ticks,host=args.host,port=args.port,case_manifest=args.case_manifest,
        reference_dir=args.reference_dir,c1_controller=args.c1_controller)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
