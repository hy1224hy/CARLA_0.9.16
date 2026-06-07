# evaluate_robustness.py 改造 — TensorBoard + 丰富评估维度

**日期**: 2026-05-17  
**状态**: ✅ 已完成  
**相关文件**: `evaluation/evaluate_robustness.py`

---

## 一、改造背景

### 改造前的问题

| 问题 | 严重程度 | 具体表现 |
|------|----------|----------|
| 无可视化监控 | **严重** | 只输出控制台文本，无法看趋势曲线，难以对比不同 checkpoint |
| 指标维度少 | **严重** | 仅有成功率、碰撞率、超时率、平均回报 4 个指标 |
| 无分 agent 分析 | **中等** | 所有 agent 的指标混在一起，某个 agent 退化无法发现 |
| 无奖励分解 | **中等** | 看不出回报变化来自哪个奖励组件 |
| 无驾驶质量指标 | **中等** | 不知道策略是"开得快但撞得多"还是"开得稳但龟速" |
| 无安全性指标 | **中等** | 碰撞率以外没有 TTC、安全距离等预警指标 |
| 结果不可复现 | **低** | 没有 seed 参数，每次运行结果略有不同 |

---

## 二、修改内容

### 2.1 新增数据类

**`EpisodeMetrics`** — 单个 episode 的完整指标记录：

```python
@dataclass
class EpisodeMetrics:
    episode_id: int                    # episode 编号
    scenario_name: str                 # 场景名
    category: str                      # 分类 (training/test/boundary)
    returns: Dict[int, float]          # 每个 agent 的回报
    total_return: float                # 总回报
    success: bool                      # 是否成功
    collision: bool                    # 是否碰撞
    timed_out: bool                    # 是否超时
    steps: int                         # episode 步数
    avg_speed_ms: float                # 平均速度 (m/s)
    max_speed_ms: float                # 最大速度 (m/s)
    avg_lateral_deviation: float       # 平均横向偏差 (m)
    max_lateral_deviation: float       # 最大横向偏差 (m)
    lane_invasions: int                # 车道偏离次数
    jerk_total: float                  # 累计 jerk 惩罚
    steer_rate_total: float            # 累计转向速率惩罚
    ttc_min: float                     # 最小 TTC（越小越危险）
    lidar_min: float                   # 最小 LiDAR 距离
    completion_time_s: float           # 完成时间（秒）
    reward_components: Dict[str,float] # 各奖励组件累加值
```

**`ScenarioSummary`** — 单个场景的汇总指标（对 N 个 episode 取均值）。

### 2.2 TensorBoard 日志集成

- 默认日志目录 `logs/eval_robustness/<YYYYmmdd_HHMMSS>/`
- 可通过 `--tb-log-dir` 自定义，`--no-tensorboard` 禁用
- 每评估完一个场景即写入（按场景索引递增 step），确保即使中途中断也有部分数据
- 全局汇总指标在 step=0 写入，方便横向对比不同 checkpoint

### 2.3 逐 step 详细数据采集

`evaluate_scenario()` 改造为每步采集：

- **速度**：从 `vehicle.get_velocity()` 获取，记录均值和最大值
- **横向偏差**：通过 `map.get_waypoint()` + `get_right_vector()` 投影计算车道中心距离
- **车道偏离**：从 `env.lane_invasion_events` 计数
- **奖励分解**：从 `info['reward_details']` 提取并累积各组件
- **安全指标**：追踪最小 TTC 和最小 LiDAR 距离值

### 2.4 新增 CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--tb-log-dir` | str | `./logs/eval_robustness` | TensorBoard 日志目录 |
| `--no-tensorboard` | flag | False | 禁用 TensorBoard |
| `--seed` | int | None | 随机种子（固定后结果可复现） |

### 2.5 增强控制台输出

每个 episode 输出扩展为：
```
第 0 轮: 成功 | 回报 142.5 | 步数 234 | 均速 32.1 km/h | 横向偏差 0.152m | 车道偏离 1次
```

最终汇总按 category 分组，额外展示：
- 分 agent 的平均回报和成功率
- 奖励组件分解表
- 各分类的均值±标准差

---

## 三、TensorBoard 指标详解

所有评价指标在 `global_step=0` 写入（用于不同 checkpoint 横向对比），记录以下标签组：

### 3.1 场景指标 (`scenario/{场景名}/`)

回答"这个场景下策略表现如何"。

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `success_rate` | 到达终点的比例 | 越高越好 |
| `collision_rate` | 碰撞比例 | 越低越好 |
| `avg_return` | 平均 episode 回报 | 越高越好 |
| `avg_speed_kmh` | 平均速度 (km/h) | 20-40，取决于场景 |
| `avg_lateral_deviation` | 平均横向偏差 (m) | < 0.3m |
| `avg_steps` | 平均 episode 步数 | 尽量短，但不要以碰撞为代价 |
| `lane_invasions_per_ep` | 每 episode 平均车道偏离次数 | 越低越好 |

### 3.2 分类汇总 (`category/{training|test|boundary}/`)

回答"在某一类场景下整体表现如何"。

指标同上，对每个 category 内所有场景做加权平均。

### 3.3 分 Agent 指标 (`agent/{0|1|2}/`)

回答"每个 agent 各自表现如何，哪个拖了后腿"。

| 指标 | 含义 |
|------|------|
| `avg_return` | 该 agent 的平均 episode 回报 |
| `success_rate` | 该 agent 单独的成功率 |
| `avg_speed_kmh` | 该 agent 的平均速度 |
| `avg_lateral_deviation` | 该 agent 的平均横向偏差 |

**使用建议**：如果某个 agent 的 `avg_return` 显著低于其他，说明该 agent 存在退化或训练不充分。对于多对抗 MAPPO 场景，合作 agent 之间的指标应该相近。

### 3.4 奖励分解 (`reward_breakdown/{组件}/`)

回答"回报的高低主要由哪些因素贡献或惩罚"。

来自 `info['reward_details']`，每个 step 自动累积，评估结束后按组件取均值：

| 指标 | 含义 | 正值驱动行为 |
|------|------|-------------|
| `progress` | 前进奖励 | 鼓励朝终点移动 |
| `speed` | 速度奖励 | 鼓励保持合理车速 |
| `centering` | 居中奖励 | 鼓励车道居中 |
| `yaw` | 航向对齐奖励 | 鼓励朝向正确 |
| `collision` | 碰撞惩罚（负值） | 惩罚碰撞 |
| `lane_invasion` | 压线惩罚（负值） | 惩罚偏离车道 |
| `step` | 每步生存惩罚（负值） | 施加时间压力 |
| `success` | 到达终点奖励 | 稀疏引导 |
| `jerk` | 急动度惩罚（v2 独有） | 鼓励平滑驾驶 |
| `steer_rate` | 转向速率惩罚（v2 独有） | 鼓励稳定转向 |
| `safety_lidar` | 安全距离（v2 独有） | 鼓励保持安全距离 |
| `ttc` | 碰撞时间（v2 独有） | 鼓励避免紧迫碰撞 |
| `time_bonus` | 时间效率奖励（v2 独有） | 鼓励快速到达 |

**使用建议**：如果 `progress` 高但 `success_rate` 低，说明 agent 在前进但不够精确；如果 `centering` 持续负向，说明策略不会车道保持；`safety_lidar` 和 `ttc` 能反映风险感知能力。

### 3.5 安全指标 (`safety/`)

回答"策略有多安全"。

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `avg_ttc_min` | 所有 episode 中最小 TTC 的均值 | 越小越危险 |
| `avg_lidar_min` | 所有 episode 中最小 LiDAR 距离的均值 | 越大越安全 |
| `collision_rate` | 整体碰撞率 | 越低越好 |

**TTC 解读**：TTC < 0 表示正在受到 TTC 惩罚（与前方障碍物 < 3 秒碰撞风险）。绝对值越大说明风险情况越严重。

### 3.6 驾驶质量指标 (`quality/`)

回答"策略开车是否平顺"。

| 指标 | 含义 | 健康范围 |
|------|------|----------|
| `avg_jerk` | 平均 jerk 惩罚（加速度变化率） | 负值越小越好（越接近 0） |
| `avg_steer_rate` | 平均转向速率惩罚 | 负值越小越好（越接近 0） |
| `avg_max_speed_kmh` | episode 内平均最高速度 | 取决于场景，过高可能危险 |

### 3.7 分布直方图 (`distribution/`)

回答"episode 间的表现差异有多大"。

| 直方图 | 含义 |
|--------|------|
| `episode_return` | 回报分布（看偏度和峰度） |
| `episode_steps` | 步数分布（看是否有大量超时 episode） |
| `avg_speed_kmh` | 速度分布（看一致性） |

**使用建议**：如果回报分布呈双峰，可能存在两类完全不同的 episode 行为模式（例如有/无碰撞）；如果步数分布在 max_steps 处堆积，说明大量 episode 无法到达终点。

---

## 四、TB 层级总览

```
scenario/{场景名}/
  success_rate / collision_rate / avg_return / avg_speed_kmh / avg_lateral_deviation / avg_steps / lane_invasions_per_ep

category/{training|test|boundary}/
  success_rate / collision_rate / timeout_rate / avg_return / avg_speed_kmh / avg_lateral_deviation / lane_invasions_per_ep

agent/{0|1|2}/
  avg_return / success_rate / avg_speed_kmh / avg_lateral_deviation

reward_breakdown/
  progress / speed / centering / yaw / collision / lane_invasion / step / success
  jerk / steer_rate / safety_lidar / ttc / time_bonus

safety/
  avg_ttc_min / avg_lidar_min / collision_rate

quality/
  avg_jerk / avg_steer_rate / avg_max_speed_kmh

distribution/
  episode_return / episode_steps / avg_speed_kmh
```

---

## 五、使用示例

```bash
cd PythonAPI/examples/mappo_training

# 基本用法
python -m evaluation.evaluate_robustness \
    --checkpoint logs/train_xxx/ckpt_1000.pt

# 完整评估 + 可视化
python -m evaluation.evaluate_robustness \
    --checkpoint logs/train_xxx/ckpt_1000.pt \
    --num-episodes-per-scenario 10 \
    --render \
    --seed 42

# 多对抗模型评估
python -m evaluation.evaluate_robustness \
    --checkpoint logs/multi_adv/ckpt_500.pt \
    --output-file results/eval_multi_adv.json

# 只测试自定义场景，禁用边界场景
python -m evaluation.evaluate_robustness \
    --checkpoint logs/train_xxx/ckpt_1000.pt \
    --test-scenario-dir data/my_test_scenarios \
    --no-include-boundary-scenarios

# 查看结果
tensorboard --logdir logs/eval_robustness/
```

---

## 六、注意事项

1. **CARLA 服务器必须先启动**：`./CarlaUE4.exe -quality-level=Low`
2. **venv 需激活**：`source venv/Scripts/activate`
3. **TensorBoard 日志目录**：每次运行创建带时间戳的子目录，不会覆盖历史记录
4. **横向偏差计算**：依赖 CARLA map waypoint 投影，仅在直道上精确
5. **TTC/LiDAR 指标**：仅 v2 奖励函数环境下有值，v1 奖励函数环境下始终为 0/inf
6. **JSON 输出**：`--output-file` 保存的 JSON 包含所有 ScenarioSummary，可用作报告数据源
