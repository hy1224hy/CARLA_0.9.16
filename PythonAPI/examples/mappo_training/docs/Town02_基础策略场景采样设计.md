# Town02 基础策略场景采样设计

## 结论

旧场景池可以支持最小实验，但不适合作为稳定的基础 MAPPO 训练基线。主要问题不是“场景数量少”，而是场景定义有状态泄漏、前后车角色偏置、方向覆盖不足，以及部分并排场景实际位于 Town02 对向车道。

## 旧设计

- `train_no_rllib_mappo_robust_v2.py` 每次更新默认收集 16 个 episode。
- classic 课程阶段为 1 / 8 / 13 个场景；4 个 `scenario_dist_*` 默认排除。
- weighted 模式先保证当前场景各出现一次，再把剩余额度按 `1 - 最近成功率 + 0.1` 分配。
- 成功率只保留每场景最近 20 次二元结果，碰撞和超时都统一记为失败。
- 最终 13 场景阶段只有 3/16 的 episode 真正受失败率权重影响。

## 旧设计的地图与采样问题

1. `scenario_02/03/04/05/07/08` 没有 `goal_zone`。旧环境不会清空上一场景目标，因此同一 JSON 的目标可能取决于随机采样顺序。
2. 原并排场景的第二辆车位于 Town02 `lane_id=-1` 一侧，却使用正向 yaw；从地图语义看属于逆向行驶，不适合作为基础合作策略的默认样本。
3. 多数场景固定 Agent 0 在后、Agent 1 在前，两个独立 Actor 容易学到 agent-id 角色偏置。
4. 几乎只覆盖北向一条路，Actor 对坐标方向和道路朝向容易过拟合。
5. 未采样场景的成功率返回 0，等价于把未知场景当作 100% 失败。
6. 原“软过渡”对累积阶段中的所有当前场景直接赋权 1，实际没有逐步引入新增场景。
7. 历史 `mappo_robust_v2_proximity_seed42` 运行在 10,000 个 episode 中仅 21 次成功、9,672 次碰撞、307 次超时；这不能单独归因于场景池，但说明当前训练基线没有形成稳定成功策略。

## 新 Town02 v2 场景池

默认目录为 `data/training_scenarios_town02_v2/`，共 12 个静态场景：

| difficulty | 数量 | 训练内容 |
|---|---:|---|
| 1 | 4 | 东/西向、无路口、中心线、长车距、交换前后角色 |
| 2 | 4 | 北向穿越路口；东/西向横向与 yaw 小扰动；角色平衡 |
| 3 | 4 | 12–14m 近距离跟驰、路口、横向与 yaw 组合扰动 |

设计约束：

- 所有出生点和目标均由 `Town02.xodr` 的 Driving waypoint 校验。
- 出生点横向偏移不超过 0.75m，yaw 与车道方向误差不超过 5°。
- Agent 0 和 Agent 1 各有 6 个场景担任后车。
- 三档各 4 个场景；课程阶段为累积的 4 / 8 / 12。
- Town02 没有同向双车道长直路，因此基础池不再伪造“同向并排”场景。需要借道/逆向博弈时，应放到对抗训练场景池并单独标注。

## 新采样规则

- 未观测场景使用 50% 成功率中性先验。
- 有历史时使用 Beta(2,2) 平滑：`posterior_success = (successes + 2) / (count + 4)`。
- 自适应优先级：`curriculum_weight × (0.25 + posterior_failure)`。
- 课程权重达到 0.5 后才进入每批强制覆盖集，避免新增场景因“每场景一次”造成概率突跳。
- 单场景设置批内次数上限，防止不可解场景吞掉整个 PPO 更新。
- TensorBoard/JSONL 新增每场景 `batch_count` 和 `priority`，可审计真实采样分布。

## 使用

从 `PythonAPI/examples/mappo_training/` 运行：

```powershell
python -m utils.validate_training_scenarios
python -m training.train_no_rllib_mappo_robust_v2 --port 2000 --town Town02
```

旧实验仍可显式使用旧池：

```powershell
python -m training.train_no_rllib_mappo_robust_v2 `
  --scenario-dir data/training_scenarios `
  --curriculum-mode classic
```

场景池指纹已经改变。不要把旧 checkpoint 直接 resume 到新场景池；新池应开一个 fresh run，旧 checkpoint 只适合作为显式的 pretrained 权重来源。
