"""Aggregate every frozen L1 development episode, retaining seed-level uncertainty."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'l1_all_cases'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Microsoft YaHei', 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 11})
COLORS = ['#4277b8', '#e08b38']
METHODS = ['independent', 'mutual']
LABELS = ['PPO', 'Mut-PPO']
data = {}
for method in METHODS:
    for seed in range(42, 47):
        folder = ROOT / 'development_audit' / f'L1_{method}_s{seed}'
        episodes = [json.loads(s) for s in (folder / 'episodes.jsonl').read_text().splitlines()]
        assert len(episodes) == 40 and len({e['case_seed'] for e in episodes}) == 40
        data[method, seed] = episodes
assert len({tuple(e['case_seed'] for e in rows) for rows in data.values()}) == 1

def seed_metric(method, field, family=None):
    return [statistics.mean(field(e) for e in data[method, seed] if family is None or e['family'] == family) for seed in range(42, 47)]

specs = [
    ('L1攻击成功率', lambda e: float(e['attack_success']), '%', 100),
    ('累计阻碍时间', lambda e: e['blocking_time'], '秒 / 回合', 1),
    ('完整换道次数', lambda e: e['completed_lane_changes'], '次 / 回合', 1),
    ('碰撞率', lambda e: float(e['collision']), '%', 100),
    ('越级率（现有判据）', lambda e: float(e['grade_violation']), '%', 100),
    ('按起始场景划分的成功率', lambda e: float(e['attack_success']), '%', 100),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
rows = []
for ax, (title, field, unit, scale) in zip(axes.flat, specs):
    if title.startswith('按'):
        for i, method in enumerate(METHODS):
            vals = [np.array(seed_metric(method, field, fam))*100 for fam in ['follow', 'adjacent']]
            x = np.arange(2) + (i-.5)*.34
            means = [v.mean() for v in vals]
            ax.bar(x, means, width=.32, color=COLORS[i], label=LABELS[i], yerr=[v.std(ddof=1) for v in vals], capsize=4, alpha=.8)
            for px, value in zip(x, means): ax.text(px, value+1.5, f'{value:.1f}', ha='center', fontsize=9)
        ax.set_xticks([0, 1], ['同车道起始', '邻道起始'])
        ax.legend(fontsize=9)
    else:
        for i, method in enumerate(METHODS):
            vals = np.array(seed_metric(method, field))*scale
            mean, sd = vals.mean(), vals.std(ddof=1)
            ax.bar(i, mean, color=COLORS[i], alpha=.8, width=.55, yerr=sd, capsize=5)
            ax.scatter(i+np.linspace(-.13,.13,5), vals, color='#243247', s=25, zorder=5)
            ax.annotate(f'{mean:.2f} ± {sd:.2f}', (i, mean+sd), xytext=(0,7), textcoords='offset points', ha='center', fontsize=10)
            rows.append(dict(metric=title, method=LABELS[i], mean=mean, sample_sd=sd, unit=unit, training_seeds=5, episodes=200))
        ax.set_xticks([0,1], LABELS)
    ax.set(title=title, ylabel=unit)
    ax.set_ylim(bottom=0)
    ax.margins(y=.25)
    ax.grid(axis='y', alpha=.2)
fig.suptitle('L1：全部独立开发评估场景的平均表现', fontsize=18)
fig.text(.5,.015,'每种方法：5个训练种子 × 相同40个场景 = 200回合；柱：种子均值；误差线：种子间样本SD；点：各训练种子。确定性执行 selected.pt。', ha='center', fontsize=10)
fig.tight_layout(rect=(0,.04,1,.95))
fig.savefig(OUT / 'L1_all_cases_overview.png', dpi=170)
fig.savefig(OUT / 'L1_all_cases_overview.pdf')
plt.close(fig)
with (OUT / 'L1_summary.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

# Time curves: no artificial padding after an episode terminates.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
time = np.arange(1,101)*.2
for i, method in enumerate(METHODS):
    ratios = np.full((5,40,100), np.nan)
    for j, seed in enumerate(range(42,47)):
        path=ROOT/'development_audit'/f'L1_{method}_s{seed}'/'trajectories.jsonl'
        traces=[json.loads(s) for s in path.read_text().splitlines()]
        for k,tr in enumerate(traces):
            for step in tr['steps']:
                index=round(step['time']/.2)-1
                ratios[j,k,index]=100*(1-step['target_speed']/max(step['reference_speed'],1e-6))
    per_seed=np.nanmean(ratios,axis=1)
    mean=np.mean(per_seed,axis=0);sd=np.std(per_seed,axis=0,ddof=1)
    axes[0].plot(time,mean,label=LABELS[i],color=COLORS[i])
    axes[0].fill_between(time,mean-sd,mean+sd,color=COLORS[i],alpha=.15)
    axes[1].plot(time,np.isfinite(ratios).sum(axis=(0,1))/200*100,label=LABELS[i],color=COLORS[i])
axes[0].set(title='目标车相对无攻击参考车的速度损失',xlabel='时间（秒）',ylabel='速度损失（%）')
axes[1].set(title='仍有有效轨迹的回合比例',xlabel='时间（秒）',ylabel='回合占比（%）',ylim=(0,105))
for ax in axes: ax.legend();ax.grid(alpha=.2)
fig.suptitle('L1：全部评估场景的平均时间曲线',fontsize=16)
fig.text(.5,.02,'阴影：种子间±1样本SD。提前终止后不补值；后段速度曲线仅统计仍在运行的回合，需结合右图解读。',ha='center',fontsize=10)
fig.tight_layout(rect=(0,.06,1,.93))
fig.savefig(OUT/'L1_all_cases_time_curves.png',dpi=170)
plt.close(fig)
print(json.dumps(rows,ensure_ascii=False,indent=2))
print('Saved figures to', OUT)
