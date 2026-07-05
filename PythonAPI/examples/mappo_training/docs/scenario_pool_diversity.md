# 场景池多样性补充：并排起点（R6）

> 日期：2026-06-04
> 涉及文件：
> - `data/training_scenarios/scenario_09~13_parallel*.json`（新增 5 个场景）
> - `training/train_no_rllib_mappo_robust_v2.py`（classic 课程 stage2 纳入 parallel）

## 1. 问题

原场景池里两车几乎都是**同一横向位置 (x)、前后排列 (不同 y)**（纵向 single-file），
唯一的横向差异场景 `06_mixed` 也是"错位"而非"并排"。**缺少真正的并排起点**
（同一 y、不同 x，左右分居两侧），导致多智能体横向交互（并排行驶、汇入、
超车冲突）训练不到。

## 2. 道路几何（**离线解析 Town02.xodr 实测**，非推测）

用 `carla.Map("Town02", xodr_string)` 离线构造地图（无需服务器）实测得到：

| 车道 | 中心 x | 宽度 | 方向 | 横向占据 |
|------|-------:|-----:|------|---------|
| **lane +1（训练用，去终点）** | -7.45 | **4.0m** | yaw +90°（+y） | `x ∈ [-9.45, -5.45]` |
| lane -1（对向，逆行） | -3.45 | 4.0m | yaw -90°（-y） | `x ∈ [-5.45, -1.45]` |

- 两车道间为 **Broken（虚线）可越中心线**；路总宽 8m，但**朝终点合法的只有 lane +1 这 4.0m**，另一条往终点方向开即逆行。
- 终点 `goal_zone` 中心 `(-7.45, 264)`。
- **关键约束**：车宽 ~2.0m，训练道仅 4.0m → **两辆车真正并排在物理上装不下**
  （需 4m+ 才有间隙）。早期 `04_left(-9.0)/05_right(-5.5)` 及并排场景**都已出道**：
  `x=-9.0` 车体左缘越路沿 0.55m，`x=-5.5` 车体右缘入逆行道 0.95m。

## 3. 并排场景（5 个，统一 `parallel` 命名）

> 设计取向（用户决策）：**保留并排、接受出道**，并加大并排间距（原 1.5m 偏挤）。
> 在 4.0m 单车道上并排必然出道，但**左右两侧风险不同**：
> - 左侧是**抬高的人行道**（spawn 风险区：`try_spawn_actor` 可能失败→自动横移 +1.5m
>   静默挪回道内，或全失败抛错崩训练）；
> - 右侧是**平坦的对向逆行道**（spawn 安全）。
>
> 故采用**非对称扩展**：左车保持已验证可 spawn 的 `-9.0`（沿用 04_left），
> 右车推进逆行道侧到 `-5.0`，车身净距由 1.5m → **2.0m**。
> 并把这些场景 **终点区加宽 `half_width 1.75→2.5`**（覆盖 `x∈[-9.95,-4.95]`），
> 让加大间距后的两车**直行即可各自到达终点**（-9.0 与 -5.0 均落在加宽终点区内）。

| 文件 | 类型 | agent0 | agent1 | dX | 终点半宽 | 出道量(车宽2.0) |
|------|------|--------|--------|---:|--------:|--------|
| `scenario_09_parallel.json` | 并排(中~64m) | (-9.0, 200) | (-5.0, 200) | 4.0 | 2.5 | A0压沿0.55 / A1逆行1.45 |
| `scenario_10_parallel_far.json` | 并排(远~88m) | (-9.0, 176) | (-5.0, 176) | 4.0 | 2.5 | A0压沿0.55 / A1逆行1.45 |
| `scenario_11_parallel_close.json` | 并排(近~24m) | (-9.0, 240) | (-5.0, 240) | 4.0 | 2.5 | A0压沿0.55 / A1逆行1.45 |
| `scenario_12_parallel_stagger.json` | 并排错位(纵错8m) | (-9.0, 198) | (-5.0, 206) | 4.0 | 2.5 | A0压沿0.55 / A1逆行1.45 |
| `scenario_13_parallel_overtake.json` | 并排超车 | (-7.45, 176) | (-5.0, 196) | 2.5 | 2.5 | A1逆行1.45 |

- 全部 `yaw=90`。09/10/11 纯并排（近/中/远）；12 并排+纵向错位；13 中央后车追逆行道侧前车。
- A1 在逆行道侧直行到加宽终点，全程不再二次跨越中心虚线 → 不会反复触发 lane_invasion 事件。
- **风险提示**：出道起点叠加 R5 加重的跨线/路沿惩罚（Curb -25）。训练时重点盯
  `lane`/`collision` 分量；若明显拖累，可（a）给这些场景设惩罚宽限步数，
  或（b）回退间距/改车道内错位。

## 3b. 俯视示意图

`docs/scenario_layout.svg`（按实测几何绘制：训练道 4m / 对向逆行道 / 路沿 / 终点区 /
车体占位 / 出道标注）。由 JSON 自动生成，改坐标后重跑生成脚本即可同步。

## 4. 课程接入（classic 模式）

`_build_default_stages` 的 stage2（中等难度）过滤条件从 `left/right`
扩展为 `left/right/parallel`：

- stage0（warmup）：`scenario_01_basic`
- stage1（中等，8 个）：basic + left/right + **5 个 parallel**
- stage2（all，13 个）：全部非 dist 场景

这样并排场景在中段课程即被采样，而非等到全场景阶段。

## 5. 校验

```bash
cd PythonAPI/examples/mappo_training
python -c "import os;from training.train_no_rllib_mappo_robust_v2 import CurriculumLearning;\
cl=CurriculumLearning(scenario_dir=os.path.join('data','training_scenarios'),mode='classic');\
[print(i,len(s)) for i,s in enumerate(cl.curriculum_stages)]"
# 期望: 0 1 / 1 8 / 2 13
```

## 6. 后续可扩展方向

- 反向并排（一车 yaw≈90、一车轻微偏航）模拟切入。
- 三横向位（需 `num_agents≥3`，当前 robust v2 为 2 智能体）。
- 不同终点 `goal_zone`（偏左/偏右）逼迫并排车择道。
