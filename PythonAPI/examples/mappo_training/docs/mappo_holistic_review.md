# MAPPO 整体复审：算法正确性 + 工程合理性

> 日期：2026-06-08
> 范围：修复 #1–#4（见 `mappo_algorithm_fix_plan.md`）后的整体复核
> 覆盖：`train_no_rllib_mappo.py` 及 4 个变体、`models/actor_critic.py`、`environment/carla_multi_agent_env.py`
>
> **历史快照说明（2026-07-23）**：本文关于 `robust_v2` “逐 episode 优势标准化”的判断已由 P0-00D 替代。当前基础策略使用逐 episode 原始 GAE + per-agent rollout 统一标准化；其他入口仍需在对应阶段迁移，不能继续引用本文“全部变体已统一/正确”的旧结论。

---

## 一、算法正确性（修复后逐项核验）

重新走查修复后的数据流，以下关键不变量全部成立：

| 检查项 | 结论 |
|--------|------|
| CTDE 结构 | actor 独立 / critic 集中；采样与训练用的 global_state 同源（同一 tick，无 staleness） ✅ |
| value 语义 | agent-i 的 value / next_value / bootstrap 均来自 `V(·, i)`；`returns = adv + V(s,i)`；critic 按 (state, agent_id) 回归各自 return，无目标冲突 ✅ |
| GAE 双 mask | `mask_value = 1 - terminated`（控价值 bootstrap）、`mask_trace = 1 - done`（控 trace）；截断保留 bootstrap、回合边界切断累积 ✅ |
| bootstrap 来源 | done 步用 reset 前对终止 state 的 `V`；rollout 尾部用最终 state 的 `V`；内部步用 `V[t+1]` ✅ |
| ratio 一致性 | 存 pre-tanh `z`、评估走 `evaluate_actions_z`，未更新策略时 ratio≡1（离线验证 dev=0.0） ✅ |
| 回合隔离 | 同一 buffer 内多 episode 拼接，done 处 trace 归零，不跨回合泄漏 ✅ |
| reset 一致性 | `prev_distances` / `reached_target` / `initial_distances` 在 reset 全部重置，无跨回合进度跳变 ✅ |
| 到达终点 agent | 奖励置 0（不累积惩罚），terminated=1 隔离 → critic 学到 V≈0 ✅ |
| 连带截断语义 | 他人碰撞致本 agent 连带结束 → terminated=0（截断）→ bootstrap，符合"本 agent 未失败" ✅ |

**离线数值验证**：4 个脚本的 `compute_gae` 均满足"截断 return=10.90（含 `1+0.99×10` bootstrap）/ 真终止 return=1.00"。`AgentConditionedCritic` 对同 state 不同 agent 输出不同 value。

**结论：标准 MAPPO 及全部变体的算法部分正确，#1–#4 已闭合且 5 个脚本语义统一。**

### 残留算法层小注（设计取舍，非 bug）
1. tanh 策略熵用 base-Normal 近似（无雅可比修正），业界通行，仅轻微稀释 `ent_coef` 探索强度。
2. 优势标准化口径不统一：主/robust/multi(v1) 整 rollout 全局标准化；robust_v2/multi_v2 逐 episode 标准化。两者皆合法，跨脚本对比时需知晓。
3. multi_adversarial v1 用 joint advantage、v2 用 per-agent advantage —— 不同团队信用分配设计，v2 才有团队内分工信号。

---

## 二、工程合理性发现（按严重度）

### 🟠 中等

**P1. 随机种子未覆盖 CARLA / TrafficManager** —— ✅ 已修复
`set_seed` 此前只播种 torch/numpy/random，env 的 `traffic_manager`（及世界他车行为）仍非确定。
**修复**：`CarlaMultiAgentEnv.__init__` 新增 `seed` 参数；创建 TrafficManager 后调用
`traffic_manager.set_random_device_seed(seed)`；各训练脚本以 `seed=args.seed` 构造 env。

**P2. 观测归一化** —— ⚠️ 原判断不准确，已修正
复核发现**观测其实已经归一化**：base 8 维 `dx/100, dy/100, distance/100, angle_diff/π, speed/30, throttle, steer, brake`，
lidar 11 维 `det_dist/ray_length ∈ [0,1]`（见 `_get_ray_detection` 第828行）。整向量已大致落在 [-1,1]。
**真正残留**：
  - 归一化常量（100.0 / 30.0 / 4.0）散落、是魔法数；
  - 位置尺度写死 100m，与实际 `straight_road_length` / 自定义场景解耦 —— 长路或越界时 dx/dy/distance 可能超出 [-1,1]；
  - 无越界保护，极端值（NaN/超范围）会直接进网络。
**修复**：把归一化常量集中为命名常量；对最终观测加一层防御性 `np.clip(obs, -CLIP, CLIP)`（CLIP 取较宽值，正常范围内零行为改变，仅兜底极端值）。保持 /100 尺度以兼容既有 checkpoint。

### 🟡 低（已记录，未改动）

| 编号 | 问题 | 说明 |
|------|------|------|
| P3 | 每个 update 强制 `env.reset()` | 丢弃 in-progress episode 重开；算法正确但浪费、轻微过采样 episode-start。可改跨 update 续跑。 |
| P4 | 动作油门/刹车互斥与梯度不一致 | `action_tanh_to_env` 把其一强制为 0，但 buffer 存原始 z → actor 对未执行的维度求梯度，brake 信号失真。可改单维纵向控制。 |
| P5 | 每步重复构造观测 | 循环顶 `get_global_state()` 与上一步 `obs_dict` 同状态重复计算（含射线投射）。可缓存复用。 |
| P6 | 已到达 agent 仍持续写 transition | reached agent 每步仍 append (r=0, terminated=1)，稀释样本。可在收集时跳过已 done 的 agent。 |
| P7 | `train_no_rlib_sharedpolicy.py` 未修复 | 仍带 #2/#3/#4（旧 compute_gae / action_tanh / atanh）。如仍在用需单独迁移，否则标注 deprecated。 |

---

## 三、总体判断

- **算法层**：正确，#1–#4 闭合，5 脚本语义统一，离线验证通过。剩余为设计取舍。
- **工程层**：无导致训练逻辑错误的硬 bug。本轮修复 P1（种子贯通）+ P2（归一化常量集中 + 防御性 clip）。P3–P6 为效率/整洁度优化，P7 待定。
- **仍需端到端验收**（需连 CARLA）：观察 TensorBoard 上首 epoch `approx_kl≈0`、`explained_variance` 转正、各 agent value 可区分。
