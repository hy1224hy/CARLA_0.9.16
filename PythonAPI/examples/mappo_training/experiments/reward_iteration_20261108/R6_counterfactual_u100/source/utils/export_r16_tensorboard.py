"""Export immutable R16 logs as explicitly retrospective TensorBoard runs."""
import hashlib
import json
from pathlib import Path
import statistics

from utils.plot_r2_budget_curves import export_tensorboard,numeric_mean,lines,read

ROOT=Path(__file__).resolve().parents[1]


def build():
    root=ROOT/'experiments/adversarial_u1000_R16_long_20261105_v1'
    review=root/'analysis/final_review_v2'
    output=root/'analysis/tensorboard_replay_20261105_v1'
    paths=[root/'runs/R2/metrics.jsonl',root/'runs/R2/episodes.jsonl',
           review/'summary.json',review/'frozen_return_and_completion.json']
    fingerprint=lambda:{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes=fingerprint();training=[];frozen=[]
    episodes=lines(paths[1]);metrics=[r for r in lines(paths[0]) if 'training_window' in r]
    for row in metrics:
        for team in ('A','B'):
            batch=[e for e in episodes if e['round']==row['completed_round'] and e['team']==team]
            w,ppo=row['training_window'][team],row['teams'][team]
            assert len(batch)==w['n'] and all(e['valid'] for e in batch)
            scalars={'return/team_discounted_mean':statistics.mean(e['team_discounted_return'] for e in batch),
                'safety/blue_bad_rate':w['b'],'critical/legacy_event_qualified_rate':w['safe_critical_event'],
                'critical/legacy_raw_event_rate':statistics.mean(e['critical_event'] for e in batch),
                'red/both_reached_rate':w['red_success_rate'],
                'value/explained_variance':ppo['explained_variance'],
                'value/critic_loss_logged_mean':numeric_mean(ppo['critic_loss']),
                'value/return_target_std':ppo['return_target_std'],
                'value/absolute_error_logged_mean':ppo['value_error_abs_mean']}
            for aid in ('2','3'):
                for field in ('mean_speed_mps','above_preferred_speed_fraction','preferred_speed_fraction',
                              'reference_lateral_p95_m','actual_steer_rate_p95_per_second','unblocked_low_speed_fraction'):
                    values=[e['driving_metrics'][aid].get(field) for e in batch]
                    values=[v for v in values if v is not None]
                    scalars['driving/blue_'+aid+'/'+field+'_episode_mean']=statistics.mean(values) if values else None
                for field in ('approx_kl','clip_frac','post_tanh_entropy','grad_norm_actor','actor_loss',
                              'surrogate_loss','entropy_bonus','deterministic_steer_mean','deterministic_longitudinal_mean'):
                    scalars['ppo/blue_'+aid+'/'+field+'_logged_mean']=numeric_mean(ppo[field+'_'+aid])
            scalars.update({'reward_component/'+k:v for k,v in w['reward_components_mean'].items()})
            training.append(dict(team=team,train_ticks=row['actual_counters']['train'],
                created_at=row['created_at'],round=row['completed_round'],n=len(batch),scalars=scalars))
    returns={(r['node'],r['team']):r['frozen_mean_discounted_return'] for r in read(paths[3])}
    for row in read(paths[2])['series']:
        q,l=row['interaction_quality'],row['legacy']
        scalars={'return/team_discounted_mean':returns[(row['target_train_ticks'],row['team'])],
            'safety/blue_event_eligible_rate':row['blue_event_eligible_count']/row['n'],
            'quality/full_driving_rate':q['driving_quality_rate_C2'],
            'quality/full_driving_and_resolved':q['joint_normal_resolved_C2'],
            'red/both_reached_rate':row['red_both_reached_count']/row['n'],
            'critical/legacy_event_qualified_rate':l['qualified_critical_rate_C2']}
        for tag,field in [('precontact_predicted_conflict_rate','predicted_conflict'),
                          ('observed_resolved_event_rate','observed_contact_free_resolved'),
                          ('censored_event_rate','contact_free_censored'),
                          ('precontact_tet_s','precontact_tet_s'),
                          ('precontact_ttc_deficit_s2','precontact_ttc_deficit_s2')]:
            scalars['critical/'+tag]=q[field+'_C2']
            scalars['reference_C1/'+tag]=q[field+'_C1']
        for key,value in row['blue_violations_overlapping'].items():scalars['safety/'+key+'_rate']=value/row['n']
        for aid,values in row['driving'].items():
            for key,value in values.items():
                if isinstance(value,dict):
                    for check,count in value.items():scalars['driving/blue_'+aid+'/'+key+'/'+check+'_rate']=count/row['n']
                else:scalars['driving/blue_'+aid+'/'+key+'_episode_mean']=value
        frozen.append(dict(team=row['team'],train_ticks=row['actual_train_ticks'],scalars=scalars))
    output.mkdir(exist_ok=False)
    validation=export_tensorboard(output/'events',training,frozen)
    assert hashes==fingerprint(),'source logs changed'
    metadata=dict(derived_from_logs=True,native_training_events=False,source_hashes=hashes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),validation=validation,
        x_axis='combined A+B actual train world ticks, not gradient steps or simulator frames',
        limitations=['A/B not independent experiment seeds','reward components are undiscounted episode totals; return is discounted',
            'frozen wall_time is export time; training wall_time comes from original log',
            'PPO scalar lists are logged epoch/minibatch aggregates, not full-rollout post-update KL',
            'driving scalar means are descriptive; per-case check failure rates determine quality'])
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'curve_data.json').write_text(json.dumps(dict(training=training,frozen=frozen),indent=2),encoding='utf-8')
    (output/'README.md').write_text('R16 72k TensorBoard历史回放。training/A、training/B各31点；frozen/A、frozen/B各4点。A/B不是独立seed。横轴为合计训练world ticks。所有事件由原始JSONL补导；训练时间来自原日志，冻结时间为导出时间。奖励分项是未折扣回合累计量，不能直接相加当作折扣回报。驾驶测量均值只作描述，资格按每个案例判定。原始证据未修改。\n',encoding='utf-8')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
