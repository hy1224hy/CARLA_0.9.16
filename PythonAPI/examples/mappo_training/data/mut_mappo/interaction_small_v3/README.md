# 小规模先后／交叉场景 v3

2026-10-29。训练24例，gate、selection、test各24例。所有文件为新的场景版本；未运行CARLA。

这里的“交叉”是同向双车道的对角错位初态。车辆不走路口交叉路线，也不由脚本强制切入；Actor决定实际动作，环境继续执行既有车道许可和目标检查。

| 类型 | 每方向 | 训练合计 | 物理含义 |
|---|---:|---:|---|
| 先后 sequential / blocking | 2 | 8 | 两辆蓝车都在对应红车前方；含完整红蓝槽位共同交换对 |
| 交叉 crossing / cut_in | 4 | 16 | 左前右后及左后右前两种对角错位，每种含编号交换对 |

以红0在左、红1在右为例，沿行驶方向向右：

```text
先后                  交叉：左前右后       交叉：左后右前
左：R0 ── B2          左：R0 ── B2        左：B2 ── R0
右：R1 ── B3          右：B3 ── R1        右：R1 ── B3
```

四方向是northbound/southbound/eastbound/westbound，仍使用Town05 Roads 44和3。镜像在道路局部左右关系下定义，实际道路不宣称是世界坐标中的精确旋转副本。编号对同时交换红0/1和蓝2/3；这不是红蓝各自独立的全部编号排列。交叉族中，每辆蓝车、每辆红车均有相等的前后角色曝光，两个交叉镜像均完整保留。镜像与编号对是相关案例，不增加独立训练seed数量。

## 共同参数与可交互性

- 两红均滚动起步2.5m/s；前方蓝2.5m/s、后方蓝3.5m/s。初速随前后物理角色移动，不与蓝编号绑定。后车初速仍在已有任务优选速度带内，没有放松速度验收阈值。
- 蓝红纵向间距按split固定为train=10m、gate=11m、selection=12m、test=13m。先后族为(+g,+g)，交叉族为(+g,-g)或(-g,+g)。不同split不是仅更换seed/名称，而有不同四车物理初态；仅是同走廊参数留出。
- 原起点在部分方向容不下后方13m的同路段出生。本版本从原并排滚动起点沿各自Driving车道前移8m，再设32m红方同车道路线。原基础与v1数据未改；移动后的红初态和目标需要新的C0行为验证。
- 后车名义相对速度为1m/s。在两车恒速、无其他响应的简化条件下，3秒保护期末最小中心距7m；从初态接近到6m中心距需4/5/6/7秒，均早于名义红车进目标区约11.6秒。这是几何筛查，不是实际无碰撞或攻击可达保证。
- 蓝车继续复用本车道对应红目标，以兼容现有环境；前蓝剩余路线19–22m、后蓝42–45m。前蓝可能较早退场，真实交互曝光和红方是否快到后蓝无法接近，仍须C0/C1检查。本次没有修改目标/退场算法或奖励。
- 时限沿用四车500步、dt=0.1s；基础模型400步的时间观测差异仍明确保留，不把严格20/2加载当驾驶资格通过。

## 采样与gate含义

复用 `first_point_complete_pairs_v1` 和既有 `sample_pairs()`。每次4回合是一个完整先后编号对加一个完整交叉编号对；方向轮换、交叉左右镜像轮换。8次采集更新的32回合覆盖全部24个训练模板，先后族部分模板重复。两族训练曝光各占一半，不按案例数量8:16直接加权。

family分别使用现有运行时支持的blocking/cut_in。它们现在对应先后与交叉布局，gate每族分别有8/16例，满足既有每族至少8例的条件。cut_in仅表示存在经地图检查的相邻车道交互机会，不证明策略实际变道或产生合格攻击。

现有评估汇总按全部24例等权，因此先后族占1/3、交叉族占2/3，与训练两族各半的采样分布不同。所有算法使用同一清单及权重，并同时报告各族结果；不能把汇总结果称为两族等权或独立物理样本均值。

`cases.json`的`preflight_subsets.smoke`登记northbound和eastbound两个方向的完整12模板子集，覆盖两条道路与全部镜像/编号对。该字段是预筛选清单，不是新增训练split；现有评估CLI不能直接传入smoke。若后续先跑该子集，应由评估API显式使用这些登记案例，不能把它当全部四方向资格。

## 离线验证与边界

`offline_map_validation.json`记录了全部96例的生产环境解析：四车出生Driving车道、初始保守车身分离、交叉族相邻车道与标线许可、四车同车道前向目标，以及每2m的非路口路径检查。使用本地Town05.xodr和`MutMAPPOEnv._resolve_case`，未创建Client，物理ticks=0。场景和地图SHA保存在报告中。

它不替代实际spawn、初速提交、碰撞回调时序、C0/C1正常驾驶或策略资格。u1000和普通蓝均仍未取得驾驶资格。

新候选实验目录：`experiments/adversarial_u1000_interactions_20261029_v2`。B0仍为外层seed42、A+B合计24k实际训练ticks，红checkpoint与SHA、20/2接口、学习参数及奖励系数不变，引用目录和案例身份是新的。没有绑定旧C1曲线，没有设置弱红豁免，没有启动真实评估或训练。

完整训练初态及gate C0/C1为(24+24)×2=96回合。可以先显式选登记的12模板做24回合工程预筛；实际运行均等待用户明确启动，并只允许一个CARLA控制器。

## 重现离线检查

```powershell
Set-Location -LiteralPath 'D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\Activate.ps1'
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m utils.build_mut_mappo_interaction_cases_v3 --validate-map --prepare-experiment --check
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m training.train_mut_mappo --config experiments/adversarial_u1000_interactions_20261029_v2/configs/B0_s42.json --validate-only
& 'D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe' -m unittest tests.test_mut_mappo_interaction_cases tests.test_first_point_experiment -v
```

生成器拒绝覆盖内容不同的已存在场景/实验。改变间距、速度、任务或分组时建立新版本，不冒充同身份续训。
