"""Seed-level partial analysis. Never writes the matrix's final report files."""
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from scipy.stats import t

root=Path(__file__).resolve().parent
study=root.parent
evidence=json.loads((root/'evidence.json').read_text(encoding='utf-8'))
audits=json.loads((root/'development_audit'/'summary.json').read_text(encoding='utf-8'))
frozen={r['run']:r for r in evidence['completed']}
groups=defaultdict(list);rows=[]
for audit in audits:
    run=audit['run'];item=frozen[run];cfg=item['config'];metrics=audit['overall']
    episodes=[json.loads(line) for line in (root/'development_audit'/run/'episodes.jsonl').read_text().splitlines()]
    violation=sum(bool(e['invalid'] or e['grade_violation'] or (e['collision'] and cfg['level']!=3)) for e in episodes)/len(episodes)
    eligible=(metrics['baseline_valid_rate']==1. and metrics['invalid_rate']<=.125
        and metrics['grade_violation_rate']<=.125 and (cfg['level']==3 or metrics['collision_rate']<=.125))
    selection=item['summary']['selection'];optimizer=item['selected_optimizer'] or {}
    selected_team=selection['team'];last=item['final_validation']['reports'][selected_team]['overall']
    row=dict(arm=item['arm'],run=run,level=cfg['level'],method=cfg['method'],seed=cfg['seed'],
        split='development',cases=metrics['cases'],execution='mean acceleration + argmax lane',
        selected_team=selected_team,selected_steps=selection['train_steps'],
        selected_beta_acceleration=optimizer.get('effective_beta_acceleration'),
        selected_beta_lane=optimizer.get('effective_beta_lane'),
        selection_asr=selection['overall']['attack_success_rate'],
        development_asr=metrics['attack_success_rate'],collision_rate=metrics['collision_rate'],
        grade_violation_rate=metrics['grade_violation_rate'],invalid_rate=metrics['invalid_rate'],
        any_constraint_violation_rate=violation,development_eligible=eligible,
        blocking_time=metrics['blocking_time'],cut_count=metrics['cut_count'],
        repeated_cut_rate=metrics['repeated_cut_rate'],completed_lane_changes=metrics['completed_lane_changes'],
        follow_asr=audit['families']['follow']['attack_success_rate'],
        adjacent_asr=audit['families']['adjacent']['attack_success_rate'],
        final_selection_asr=last['attack_success_rate'],
        late_lineage_mean=item['late_selected_lineage']['mean'],
        late_lineage_drawdown=item['late_selected_lineage']['drawdown'],
        checkpoint_sha256=audit['checkpoint_sha256'])
    rows.append(row);groups[(cfg['level'],cfg['method'])].append(row)
with (root/'results_partial.csv').open('w',newline='',encoding='utf-8-sig') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def stats(values):
    return dict(n=len(values),mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values)>1 else None)

def fmt(values,percent=True):
    result=stats(values);scale=100 if percent else 1
    value=f'{result["mean"]*scale:.2f}'
    value+=f' ± {result["sd"]*scale:.2f}' if result['sd'] is not None else '（仅1 seed）'
    return value+('%' if percent else '')

group_stats=[]
for (level,method),items in sorted(groups.items()):
    group_stats.append(dict(level=level,method=method,seeds=[r['seed'] for r in items],
        metrics={key:stats([r[key] for r in items]) for key in ('development_asr','collision_rate',
            'grade_violation_rate','any_constraint_violation_rate','blocking_time','cut_count',
            'follow_asr','adjacent_asr','late_lineage_drawdown','late_lineage_mean')}))
a={r['seed']:r for r in groups[(1,'independent')]}
b={r['seed']:r for r in groups[(1,'mutual')]}
seeds=sorted(a.keys()&b.keys());differences=[b[s]['development_asr']-a[s]['development_asr'] for s in seeds]
delta=stats(differences)
half=float(t.ppf(.975,len(seeds)-1))*delta['sd']/math.sqrt(len(seeds))
paired=dict(seeds=seeds,differences=differences,mean=delta['mean'],
    sd=delta['sd'],t_interval_95=[delta['mean']-half,delta['mean']+half],
    note='Exploratory paired t interval across training seeds, small n=5; development, not final test')
event_counts=Counter()
for item in evidence['completed']:
    if item['config']['level']==2 and item['config']['method']=='independent':
        event_counts.update(item['training_episodes']['diagnostics'])
summary=dict(snapshot_time_beijing=evidence['snapshot_time_beijing'],groups=group_stats,
             L1_paired_development=paired,L2_independent_event_counts=dict(event_counts))
(root/'statistics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')

lines=['# 混合动作批量实验阶段性分析','',
    f'冻结快照：{evidence["snapshot_time_beijing"]}。本报告分析该快照已完成的16个模型，不是55组完整结论。','',
    '## 1. 完成状态与证据范围','',
    '- 冻结配置55个；完成16个、日志持续更新2个、尚未启动37个。',
    '- main：L1两方法×5种子已完成；L2 independent五种子完成，mutual仅s42完成；L3尚未启动。',
    '- beta、reward尚未启动，不能分析敏感性或奖励消融结果。',
    '- 原 audit_selected 与根目录 report.md 尚未生成，原因是编排器要求全部55组完成后才审计。',
    '- 本次复制并冻结16个已完成 selected.pt，在独立开发集重新评估，每模型40场景（follow/adjacent各20）。',
    '- 共640个模型评估回合，但算法统计独立单位仍是训练种子；不把场景回合当成独立训练重复。',
    '- 已完成运行来源摘要一致，独立审计通过当前源码和运行版本检查；未修改训练代码、配置或编排进度。',
    '- 执行模式为均值加速度＋argmax车道。最终test未使用。','',
    '## 2. 独立开发评估','',
    '均值±样本标准差（ddof=1），以独立训练种子聚合。','',
    '| 等级/方法 | seeds | 攻击成功率 | 碰撞率 | 越级率 | 任一约束违规率 | follow成功率 | adjacent成功率 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for key,items in sorted(groups.items()):
    lines.append('| L'+str(key[0])+' '+key[1]+' | '+str(len(items))+' | '+' | '.join(
        fmt([r[k] for r in items]) for k in ('development_asr','collision_rate','grade_violation_rate',
            'any_constraint_violation_rate','follow_asr','adjacent_asr'))+' |')
lines+=['','L2 mutual只有一个已完成种子，不能与五种子independent作算法优劣比较；L3没有本批数据。','',
    '## 3. L1：可学习，但互学习未显示总体优势','',
    f'- independent开发ASR：{fmt([r["development_asr"] for r in groups[(1,"independent")]])}。',
    f'- mutual开发ASR：{fmt([r["development_asr"] for r in groups[(1,"mutual")]])}。',
    f'- 配对差（mutual−independent）均值：{100*delta["mean"]:.2f}个百分点；五种子中4个下降、1个上升。',
    f'- 探索性配对t区间：[{100*paired["t_interval_95"][0]:.2f}, {100*paired["t_interval_95"][1]:.2f}]个百分点，包含0；n=5且依赖分布假设，不宜夸大统计确定性。',
    '- mutual的跨种子ASR标准差较小，但均值也更低，不能单独把SD下降解释为综合性能改善。',
    '- mutual碰撞均值高于independent，越级均值略低；应联合报告成功与约束，不选择性展示某一列。','',
    '| seed | independent ASR | mutual ASR | 配对差（百分点） | independent约束违规 | mutual约束违规 |',
    '|---|---:|---:|---:|---:|---:|']
for seed in seeds:
    left,right=a[seed],b[seed]
    lines.append(f'| {seed} | {left["development_asr"]:.1%} | {right["development_asr"]:.1%} | '
        f'{100*(right["development_asr"]-left["development_asr"]):+.1f} | '
        f'{left["any_constraint_violation_rate"]:.1%} | {right["any_constraint_violation_rate"]:.1%} |')
lines+=['','按现有baseline有效、无效/越级/碰撞各≤12.5%的门槛，在开发集重新检查：']
for method in ('independent','mutual'):
    items=groups[(1,method)]
    lines.append(f'- L1 {method}：{sum(r["development_eligible"] for r in items)}/{len(items)} 个模型仍通过。')
lines+=['','这些门槛在40场景下一个样本对应2.5个百分点；边界附近资格受小样本波动影响，不能理解为安全保证。',
    'L1平均完整换道次数也应检查：模型可能通过频繁换道维持温和影响，不应未经轨迹确认就全部称为纯慢速跟车阻碍。','',
    '## 4. L2：主要瓶颈是时距资格','',
    f'- 五种子independent开发ASR：{fmt([r["development_asr"] for r in groups[(2,"independent")]])}。',
    '- 种子结果不一致，两个开发评估为0，s46较高；不能视为稳定收敛。','',
    '| 训练事件计数（五个independent合计） | 数量 |', '|---|---:|']
for key in ('candidates','geometric','thw_ok','thw_miss','react_hit','react_miss'):
    lines.append(f'| {key} | {event_counts[key]} |')
lines+=[f'',f'THW合格/几何确认 = {event_counts["thw_ok"]}/{event_counts["geometric"]} = {event_counts["thw_ok"]/event_counts["geometric"]:.2%}；'
    f'反应确认/THW合格 = {event_counts["react_hit"]}/{event_counts["thw_ok"]} = {event_counts["react_hit"]/event_counts["thw_ok"]:.2%}。',
    '这些是状态机事件计数的条件比例，不是独立回合成功率，也不是完全闭合的因果漏斗。',
    '目前换道和几何对齐已经大量发生，继续扩大随机性不直接解决主要门槛。应在新开发版本记录跨道时THW、完成对齐时THW、首次制动时间、候选取消原因，核验目标提前制动是否使确认时距离开窗口。',
    '不建议直接放宽0.3～1.2 s门槛来制造成功。先确认任务定义中的“切入时距”应在哪个事件时刻测量，再做冻结版本对照。',
    '反应确认几乎必过也提示需要检查是否复用了先前速度损失，重复切入的新增反应因果性仍需轨迹核验。','',
    '## 5. L2 mutual s42的成功不能归因于互学习','',
    '该模型选在20480合计步；对应更新的实际两个beta均为0。调度按更新开始步计算，此轮仍处于warmup。',
    '因此它的开发成功率15%只是互学习启用前PPO检查点的表现；不能用它高于同seed independent的7.5%证明互学习有效。后者选于440320步，是不同检查点。',
    '其余mutual种子尚未完成，完整对比应等待全部五种子冻结审计。','',
    '## 6. 收敛性与稳定性','',
    '高分selected.pt不能替代训练稳定性证据。例：L1 independent s42从420864步选中的90.625%验证ASR降到500000步同副本的0%。',
    '同时，一些末期原始ASR高于selected ASR，但碰撞或越级超过资格门槛，因此没有被选中；不能只凭ASR大小判断选模错误。',
    '分析末期同一副本谱系，保留约束曲线；不能只画A/B逐轮择优包络。归零后保持平坦也不能算有效策略稳定。','',
    '| L1方法 | 后25%预算同谱系ASR均值 | 后25%最大回撤均值 |', '|---|---:|---:|']
for method in ('independent','mutual'):
    items=groups[(1,method)]
    lines.append(f'| {method} | {fmt([r["late_lineage_mean"] for r in items])} | '
                 f'{fmt([r["late_lineage_drawdown"] for r in items])} |')
lines+=['','这两项只描述selection时间序列，不能独立证明跨场景鲁棒性。','',
    '## 7. 评估和报告口径需要改正/明确','',
    '1. 当前冻结评估为确定性执行，PPO训练为随机采样。此前pilot已出现随机执行有碰撞、argmax不换道的情况；那是另一批数据，不能移植为本批结论。建议预先确定策略池是否采用随机执行，并将随机/确定性作为分开的评估结果，不能临时择优报告。',
    '2. matrix_report.py使用statistics.pstdev（ddof=0）。若将5个训练种子视为独立重复并报告样本SD，应改为stdev（ddof=1）；n=5时前者比样本SD约小10.6%。本报告统一使用样本SD。',
    '3. no_dense配置实际仅令effect_weight=0，保留risk_weight对应的密集时距项和控制代价。准确名称应为“去速度损失/影响项”，不能解释成去掉所有密集奖励。',
    '4. no_event同时去掉有效切入事件奖金与终局成功奖金，是联合消融；不能单独归因于其中一项。',
    '5. L1应同时看分级成功和约束，不能只把blocking_time作为首要结论；失败前积累的阻碍时间仍可能很长。',
    '6. control/monitor.md停留在启动初期的0/8，已经不能代表当前进度。应以实际summary/metrics与本次冻结清单为准；本报告未擅自重启或修改编排。',
    '7. 标签beta/reward虽有配置，但尚无训练结果。缺失不得填0，也不能制作完成的消融图。','',
    '## 8. 当前可支持与不可支持的研究结论','',
    '- 可支持：混合动作下L1可学习，在本开发集部分种子表现很强；L2已有少量有效事件，但整体成功率低且波动大。',
    '- 可支持：L1互学习在当前beta与确定性执行口径下表现为较低均值、较小跨种子SD，未显示综合优势。',
    '- 尚不支持：互学习普遍提高三级策略性能或鲁棒性；完整三级分化及策略池覆盖；beta最优值；各奖励分量有效性。',
    '- L3主实验未运行，不能据本矩阵讨论其结果。',
    '- 所有结果均为开发评估，不是保留test的最终结果，也不是CARLA迁移证据。','',
    '## 9. 建议的下一步','',
    '1. 保持当前运行的配置和源码冻结，先补齐main的五种子对照；当前不因局部负结果中途修改已定义实验。',
    '2. 后续投入消融前，明确论文主执行口径；若研究随机策略池，另外预注册固定场景×多个动作采样seed的审计，保留确定性结果作为补充。',
    '3. 在独立开发版本诊断L2事件时序；必要时修订“切入时距”的测量时刻，旧/新协议分别统计。',
    '4. beta和奖励消融完成后，按实际移除内容解释；展示逐种子点、配对差值及任务失败率。',
    '5. 使用冻结后的最终模型在保留test与目标参数扰动上验证；仅训练曲线或SD下降不足以证明鲁棒性。','',
    '## 10. 文件','',
    '- [逐模型独立审计总表](development_audit/report.md)',
    '- [种子级长表CSV](results_partial.csv)',
    '- [统计明细JSON](statistics.json)',
    '- [冻结证据清单](evidence.json)',
    '- 原始训练运行位于本目录上一级的 train/main/；正式55组汇总仍由原编排器负责。']
(root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Export a paired scientific figure; no generated graphic is used as data.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,key,title in ((axes[0],'development_asr','L1 development attack success'),
                     (axes[1],'any_constraint_violation_rate','L1 development constraint violations')):
    for seed in seeds:
        ax.plot([0,1],[a[seed][key],b[seed][key]],marker='o',alpha=.65,label=str(seed))
    ax.set_xticks([0,1],['Independent','Mutual']);ax.set_ylim(-.025,1.025)
    ax.set_ylabel('Rate');ax.set_title(title);ax.grid(alpha=.2)
axes[1].legend(title='Training seed',fontsize=8)
fig.tight_layout();fig.savefig(root/'L1_paired_development.png',dpi=180);plt.close(fig)
print(json.dumps(dict(groups=group_stats,paired_L1=paired,L2_events=dict(event_counts)),indent=2))
