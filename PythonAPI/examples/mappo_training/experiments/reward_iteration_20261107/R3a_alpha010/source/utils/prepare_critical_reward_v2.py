"""Prepare the next single-seed blue-only pilot without contacting CARLA."""
import argparse
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def prepare(root):
    from training.checkpoint_protocol import atomic_write_json, sha256_file
    from training.mut_mappo_config import load_config, portable_config, validate_config
    from environment.mut_mappo_critical_reward_v2 import PROFILE, EXTRA_DEFAULTS
    previous=ROOT/'experiments/adversarial_u1000_critical_reward_20261101_v3'
    old_state=json.loads((previous/'control/pipeline_status.json').read_text(encoding='utf-8'))
    if old_state['status']!='COMPLETED_FOR_ANALYSIS':
        raise RuntimeError('finish the original frozen comparison before creating the next pilot')
    config=portable_config(load_config(previous/'configs/R1_B0_s42.json'))
    config['task']['reward_profile']=PROFILE
    config['task']['team_reward'].update(EXTRA_DEFAULTS, critical_weight=3., pace_weight=.1,
        reference_dir=str(root/'references/R2'))
    config['diagnostic_authorization']['source']='User authorized iterative blue reward adjustment and clarified failed-C1 weak-red engineering scope on 2026-11-01'
    validated=validate_config(config)
    baseline={}
    for group in ('R0','R1'):
        marker=json.loads((previous/'control'/f'{group}_gate_completed.json').read_text(encoding='utf-8'))
        for name,digest in marker['artifact_hashes'].items():
            if sha256_file(previous/name)!=digest:raise ValueError('baseline report changed')
        baseline[group]=copy.deepcopy(marker)
    root.mkdir(parents=True,exist_ok=False)
    (root/'configs').mkdir();(root/'control').mkdir()
    atomic_write_json(config,root/'configs/R2_B0_s42.json')
    authorization=json.loads((previous/'control/training_authorization.json').read_text(encoding='utf-8'))
    authorization['scope']='Versioned follow-up to the authorized weak-red reward adjustment; seed42 B0 R2, 24000 train world ticks'
    atomic_write_json(authorization,root/'control/training_authorization.json')
    atomic_write_json(validated['_root_validation'],root/'control/offline_compatibility.json')
    atomic_write_json(dict(status='PREPARED_OFFLINE',group='R2',profile=PROFILE,
        predecessor=str(previous.relative_to(ROOT)),baseline_reports=baseline,
        baseline_comparison='analysis/reward_comparison_complete_v1/comparison.json',
        train_tick_budget=24000,reference_episodes=144,frozen_gate_episodes=144,
        outer_seed=42,independent_seed_count=1,team_count=2,
        reset_from='Same frozen red Actor initialization; fresh blue optimizers, four-car Critic and ValueNorm',
        changed_variables='Reward profile and its explicitly versioned credit grace only; no PPO, initial-state or threshold changes',
        reward_changes=dict(exposure_budget=3.,primary_persistent_event_bonus=5.,
            primary_event_duration_s=.3,primary_event_ttc_s=[.5,1.5],credit_grace_s=1.,
            normal_overspeed_total_scale=10.,normal_comfort_total_scale=5.,
            legally_exempted_lane_cost_scale=10.,pace_weight=.1),
        measurement_protocol_unchanged=True,test_used=False,
        advancement_rule='Do not call success from return or G alone. Review the unchanged bad-rate screen, dangerous interactions, speed/lane/control and paired C1 contrasts together.',
        failure_rule='Preserve failed outcomes and costs; never lower acceptance thresholds or silently extend budget',
        cost_scope='Training setup included; internal and independent evaluation, references and abandoned attempts counted separately',
        estimated_wall_time_note='Prior runs suggest roughly 1-2 hours including references, blue training and frozen gate; not a deadline guarantee',
        source_sha256=sha256_file(ROOT/'environment/mut_mappo_critical_reward_v2.py')),
        root/'control/experiment_plan.json')
    return validated['_root_validation']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('experiments/adversarial_u1000_critical_reward_20261101_v4'))
    args=parser.parse_args();root=args.root if args.root.is_absolute() else ROOT/args.root
    print(json.dumps(prepare(root.resolve()),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
