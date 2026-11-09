"""File-only descriptive review of the registered real-rollout diagnosis."""
import json
import statistics
from pathlib import Path


def build(root):
    from utils.r2_budget_experiment import records,read
    from training.checkpoint_protocol import atomic_write_json as write,sha256_file
    output=root/'analysis';output.mkdir(exist_ok=True)
    summary={'scope':'single-seed engineering diagnosis; independent cases, A/B not seeds',
        'groups':{},'frozen':[],'sources':{}}
    for group in ('N4','R4','R16'):
        updates=records(root/'audits'/group/'updates.jsonl')
        metrics=[r for r in records(root/'runs'/group/'metrics.jsonl') if 'training_window' in r]
        audit_roles=[v for row in updates for v in row['roles'].values()]
        g={'updates_audited':len(updates),'roles_audited':len(audit_roles),
           'max_action_error':max(r['max_action_error'] for r in updates),
           'max_logp_error':max(r['max_logp_error'] for r in updates),
           'max_gae_error':max(v for r in updates for v in r['gae_errors'].values()),
           'positive_surrogate_fraction':statistics.mean(r['surrogate_improvement']>0 for r in audit_roles),
           'positive_advantage_alignment_fraction':statistics.mean(r['advantage_logp_alignment']>0 for r in audit_roles),
           'mean_entropy_to_policy_grad_ratio':statistics.mean(r['entropy_to_policy_grad_ratio'] for r in audit_roles),
           'costs':metrics[-1]['actual_counters'],'training':{}}
        for team in ('A','B'):
            episodes=[r for r in records(root/'runs'/group/'episodes.jsonl') if r['team']==team]
            n=min(48,len(episodes)//2)
            g['training'][team]={label:dict(n=len(rows),mean_return=statistics.mean(r['team_discounted_return'] for r in rows),
                bad_rate=statistics.mean(not r['blue_eligible'] for r in rows))
                for label,rows in [('first',episodes[:n]),('last',episodes[-n:])]}
        summary['groups'][group]=g
        for node in read(root/'control'/(group+'_nodes.json')):
            path=root/'evaluations'/(group+'_frozen_'+str(node['target']).zfill(6))/'analysis/critical_report.json'
            report=read(path);summary['sources'][str(path)]=sha256_file(path)
            if any(r['measurement_coverage']!=1 for r in report['team_rows']):raise ValueError('incomplete frozen evidence')
            for q,legacy in zip(report['interaction_quality_team_rows'],report['team_rows']):
                if q['team']!=legacy['team']:raise ValueError('team ordering differs')
                summary['frozen'].append(dict(group=group,node=node,quality=q,legacy=legacy))
    write(summary,output/'summary.json')
    lines=['# 三步真实训练诊断结果','',
        '所有组固定弱红u1000。N4仅关闭对抗收益及对应严重险情项，保留现有正常驾驶成本；它不等于历史双车基础训练。',
        '回报只比较同一奖励组的前后变化；组间不同奖励的绝对回报不能直接排序。',
        '真实采样检查与同批代理目标改善是必要证据，不能证明新策略在环境中更好。',
        '','|组|队|训练ticks|蓝方事件资格率|接触前持续冲突率|局部无接触解除率|完整驾驶质量|完整驾驶且解除险情|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x:'未知' if x is None else f'{100*x:.1f}%'
    for row in summary['frozen']:
        q,l=row['quality'],row['legacy']
        lines.append('|'+ '|'.join([row['group'],q['team'],str(row['node']['actual']),fmt(l['blue_eligible_rate_C2']),
            fmt(q['predicted_conflict_C2']),fmt(q['observed_contact_free_resolved_C2']),fmt(q['driving_quality_rate_C2']),fmt(q['joint_normal_resolved_C2'])])+'|')
    lines.extend(['','衡量成功需分层：',
        '1. 工程正确性：动作、log概率、GAE、终止/bootstrap和团队尾项检查通过。',
        '2. 学习有效性：固定案例下同奖励回报、蓝方驾驶质量改善；PPO代理目标改善只能支持更新方向。',
        '3. 对抗有效性：配对C1/初始模型后，在蓝方完整正常驾驶前提下，出现更多接触前持续危险交互并观测到无接触解除；碰撞、越界、违规收益单列。',
        '4. 研究有效性：独立seed与未用于调参的场景复验。此处24例只能诊断，不能宣称正式算法优越性。',
        '', '因果解释边界：N4与R4差异检验对抗奖励组合；R4与R16检验采样批量及其伴随更新频率。不能把这些差异直接归因于单个奖励项。',
        'A/B各24例，结果不能当作48个独立实验seed。C1完整驾驶未知时不得填0或1。',
        '', '优化审计：','```json',json.dumps(summary['groups'],ensure_ascii=False,indent=2),'```'])
    (output/'review.md').write_text('\n'.join(lines),encoding='utf-8')
    return summary
