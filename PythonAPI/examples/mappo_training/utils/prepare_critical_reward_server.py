"""Portable offline preparation for the latest R2 blue reward server pilot.

No local experiment predecessor, authorization marker, CARLA connection or training.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/'configs/mut_mappo/critical_reward_v2_server.json'


def resolve(path):
    path=Path(path).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def prepare(checkpoint, output_dir, *, source_config=None, expected_sha256=None,
            device='cuda', seed=42, train_ticks=None, host='127.0.0.1', port=2000,
            case_manifest=None, reference_dir=None, c1_controller=None,
            lr_comparison=False, updates=None, episodes_per_team_update=None,
            eval_every_updates=100):
    from training.checkpoint_protocol import load_checkpoint_file, atomic_write_json, sha256_file
    from training.mut_mappo_config import load_config, portable_config, validate_config
    from environment.mut_mappo_critical_reward_v2 import PROFILE
    output=resolve(output_dir)
    if output.exists():raise FileExistsError('output exists; choose a new experiment directory')
    if type(lr_comparison) is not bool:
        raise ValueError('lr_comparison must be a boolean')
    update_mode=updates is not None
    if update_mode:
        if type(updates) is not int or updates<=0:
            raise ValueError('updates must be a positive integer of complete A+B rounds')
        if lr_comparison or train_ticks is not None:
            raise ValueError('--updates cannot be combined with --lr-comparison or --train-ticks')
        if episodes_per_team_update is None:episodes_per_team_update=16
        if (type(episodes_per_team_update) is not int or episodes_per_team_update<2
                or episodes_per_team_update%2):
            raise ValueError('episodes per team update must be a positive even number for complete pairs')
        if type(eval_every_updates) is not int or eval_every_updates<=0:
            raise ValueError('eval every updates must be a positive integer')
    else:
        if episodes_per_team_update is not None:
            raise ValueError('--episodes-per-team-update requires --updates')
        if train_ticks is None:train_ticks=60000 if lr_comparison else 24000
    if ((not update_mode and (type(train_ticks) is not int or train_ticks<=0))
            or type(seed) is not int):
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
    if update_mode and (config.get('method')!='off' or config.get('algorithm','mappo')!='mappo'
            or config.get('team_count',2)!=2):
        raise ValueError('R16 update training requires source method=off, algorithm=mappo and team_count=2')
    if lr_comparison and (type(config.get('learning',{}).get('ppo_epochs')) is not int
            or config['learning']['ppo_epochs']!=2
            or type(config.get('run',{}).get('episodes_per_team_update')) is not int
            or config['run']['episodes_per_team_update']!=4):
        raise ValueError('B0 LR comparison requires source learning.ppo_epochs=2 and '
                         'run.episodes_per_team_update=4; source values are not overwritten')
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
    entries=[]
    entry_configs=[]
    if update_mode:
        config.update(method='off',algorithm='mappo',team_count=2)
        config['mutual']['quality_metric']='qualified_primary_event_credit'
        config['evaluation']['selection_metric']='qualified_primary_event_credit'
        config['run'].update(stop_rule='rounds',schedule_clock='rounds',rounds=updates,
            lr_horizon_rounds=updates,lr_schedule='linear',lr_min_factor=0.1,
            train_tick_budget=None,episodes_per_team_update=episodes_per_team_update)
        for key in ('lr_horizon_train_ticks','lr_decay_train_ticks'):
            config['run'].pop(key,None)
        run_id=f'R2_B0_s{seed}'
        entries.append(dict(run_id=run_id,group='R2_B0',seed=seed,
            config=str(output/'config.json'),run_dir=str(output/'runs'/run_id)))
    elif lr_comparison:
        config.update(method='off',algorithm='mappo',team_count=2)
        config['mutual']['quality_metric']='qualified_primary_event_credit'
        config['evaluation']['selection_metric']='qualified_primary_event_credit'
        config['run'].pop('lr_horizon_train_ticks',None)
        config['run'].update(stop_rule='train_ticks',schedule_clock='train_ticks',
            lr_schedule='linear',lr_decay_train_ticks=train_ticks,lr_min_factor=0.1)
        for schedule in ('linear','constant'):
            candidate=deepcopy(config)
            candidate['run']['lr_schedule']=schedule
            if schedule=='constant':
                candidate['run'].pop('lr_decay_train_ticks',None)
                candidate['run'].pop('lr_min_factor',None)
            group='B0_'+schedule.upper()
            run_id=f'{group}_s{seed}'
            entries.append(dict(run_id=run_id,group=group,seed=seed,
                config=str(output/'configs'/(run_id+'.json')),
                run_dir=str(output/'runs'/run_id)))
            validate_config(candidate)
            entry_configs.append(candidate)
    checked=validate_config(config)
    # All checks above are read-only; never create output for a rejected upload/config.
    output.mkdir(parents=True,exist_ok=False)
    atomic_write_json(config,output/'config.json')
    if lr_comparison:
        for entry,candidate in zip(entries,entry_configs):
            atomic_write_json(candidate,Path(entry['config']))
        experiment=dict(schema='first_point_experiment_v1',suite='B0_lr_comparison',
            base_config=str(output/'config.json'),entries=deepcopy(entries),
            train_tick_budget=train_ticks,status='PREPARED_NOT_RUN',
            diagnostic='matched_B0_linear_vs_constant_learning_rate',
            execution='one user-started CARLA, serial; A/B cost combined; complete-boundary soft cap',
            reward='source R2 reward and safety thresholds unchanged; shared C1 references',
            independent_unit='outer training seed; no A/B pooling',
            comparison_fields=['run.lr_schedule','run.lr_decay_train_ticks','run.lr_min_factor'],
            carla_connected=False,training_launched=False)
        atomic_write_json(experiment,output/'experiment.json')
    elif update_mode:
        experiment=dict(schema='first_point_experiment_v1',suite='R2_updates',
            base_config=str(output/'config.json'),entries=deepcopy(entries),
            train_tick_budget=None,target_updates=updates,eval_every_updates=eval_every_updates,
            stop_rule='rounds',status='PREPARED_NOT_RUN',
            update_semantics='one complete A+B round; each team performs one PPO update',
            execution='one user-started CARLA, serial; ticks audited without a stopping cap',
            reward='source R2 reward and safety thresholds unchanged',
            independent_unit='outer training seed; no A/B pooling',
            carla_connected=False,training_launched=False)
        atomic_write_json(experiment,output/'experiment.json')
    record=dict(status='PREPARED_NOT_TRAINED',schema='critical_reward_server_preparation_v1',
        profile=PROFILE,config=str(output/'config.json'),
        run_dir=None if lr_comparison else entries[0]['run_dir'] if update_mode else str(output/'run'),
        source_config=str(source),checkpoint=str(checkpoint),red_sha256=digest,
        source_update=payload.get('update'),device=device,seed=seed,train_tick_budget=train_ticks,
        reference_dir=config['task']['team_reward']['reference_dir'],
        case_manifest=config['case_manifest'],c1_controller=config['task'].get('c1_controller','frozen_actor'),
        config_sha256=sha256_file(output/'config.json'),validation=checked['_root_validation'],
        scope='weak_red_engineering_reward_pilot_not_formal_driving_qualification',
        initialization='fresh blue learners from supplied base actors; not resume of an old reward run',
        carla_connected=False,training_launched=False)
    if lr_comparison:
        record.update(lr_comparison=True,
            diagnostic='matched_B0_linear_vs_constant_learning_rate',
            experiment=str(output/'experiment.json'),entries=deepcopy(entries),
            comparison_fields=experiment['comparison_fields'],
            matching='same red checkpoint, source actors, seed, task, C1 references, learning parameters and train tick budget')
    elif update_mode:
        record.update(target_updates=updates,eval_every_updates=eval_every_updates,
            episodes_per_team_update=episodes_per_team_update,stop_rule='rounds',
            update_semantics=experiment['update_semantics'],
            experiment=str(output/'experiment.json'),entries=deepcopy(entries),
            learning_rate='linear by complete update rounds with a 0.1 floor',
            tick_policy='audit_only_no_tick_stopping_cap')
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
    parser.add_argument('--train-ticks',type=int,
        help='training ticks per run (default: 24000; 60000 with --lr-comparison)')
    parser.add_argument('--lr-comparison',action='store_true',
        help='prepare matched B0 linear and constant learning-rate diagnostic runs')
    parser.add_argument('--updates',type=int,
        help='complete A+B update rounds; each team updates this many times, without a tick cap')
    parser.add_argument('--episodes-per-team-update',type=int,
        help='episodes per team update with --updates (default: 16; positive even number)')
    parser.add_argument('--eval-every-updates',type=int,default=100,
        help='selection evaluation interval recorded for update-mode runs (default: 100)')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=2000)
    parser.add_argument('--case-manifest',type=Path)
    parser.add_argument('--reference-dir',type=Path)
    parser.add_argument('--c1-controller',choices=('frozen_actor','lane_pid_v1'))
    args=parser.parse_args()
    result=prepare(args.checkpoint,args.output_dir,source_config=args.source_config,
        expected_sha256=args.expected_sha256,device=args.device,seed=args.seed,
        train_ticks=args.train_ticks,host=args.host,port=args.port,case_manifest=args.case_manifest,
        reference_dir=args.reference_dir,c1_controller=args.c1_controller,
        lr_comparison=args.lr_comparison,updates=args.updates,
        episodes_per_team_update=args.episodes_per_team_update,
        eval_every_updates=args.eval_every_updates)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
