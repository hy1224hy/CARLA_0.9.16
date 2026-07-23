# MAPPO GAE、优势标准化与 PPO 更新原理

> 用途：解释本项目 P0-00D 的数学口径、工程实现和验收方法，可作为毕业论文“回报估计与策略优化”部分的写作素材。
>
> 当前状态：P0-00D 已于 2026-07-23 完成 `train_no_rllib_mappo_robust_v2.py` 的 synthetic/unit 验收；真实 CARLA smoke 留在 P0-00G。本文不能替代真实仿真性能结论。

## 1. 本阶段解决的三个归属问题

MAPPO 训练中，每条有效 transition 必须同时具有三个清晰归属：

1. **agent 归属**：局部观测、动作、旧 log-prob、奖励、value 和 advantage 必须属于同一 agent；
2. **episode 归属**：GAE trace 不得跨 episode 递推；
3. **优化目标归属**：Actor 使用标准化 advantage，Critic 使用原始奖励尺度的 return。

当前工程保持两个独立 Actor；centralized critic 共享参数，但输入 agent ID，输出 `V(s, i)`。共享 Critic 不等于共享团队 return：每个 agent 仍使用自己的 reward、GAE 和 return。

## 2. 单 episode 的原始 GAE

对 agent \(i\) 的某个 episode，时间差分残差为：

$$
\delta_t^i =
r_t^i + \gamma(1-T_t^i)V(s_{t+1},i)-V(s_t,i)
$$

GAE 为：

$$
A_t^i =
\delta_t^i + \gamma\lambda(1-D_t^i)A_{t+1}^i
$$

其中：

- \(T_t^i\) 对应 `terminated`；
- \(D_t^i\) 对应 `done = terminated or truncated`；
- `mask_value = 1 - terminated`：真终止切断 value bootstrap；
- `mask_trace = 1 - done`：终止和截断都切断跨 episode trace。

因此：

| 边界 | 最后一步 value target | GAE trace |
|---|---|---|
| `terminated=True` | 不使用最终状态 value | 切断 |
| `truncated=True` | 使用同帧最终状态 `V(s_T,i)` | 切断 |

P0-00D 的 `compute_raw_gae_per_episode()` 要求边界连续、无重叠地覆盖全部 transition，并要求 bootstrap 数量和 episode 数量完全一致。自然终止 episode 若携带非零 bootstrap 会直接报错。

## 3. 为什么不能逐 episode 标准化

假设两个 episode 的原始 advantage 为：

```text
成功 episode：[20, 10]
普通 episode：[ 2,  1]
```

若分别标准化，两者都会近似变为：

```text
[1, -1]
```

这使用了两套不同的仿射变换，抹掉了不同结局之间的相对尺度，还可能放大短 episode 的微小方差。正确方式是先独立计算每个 episode 的原始 GAE，再按 agent 拼接：

```text
agent i:
  episode 1 raw advantage
  episode 2 raw advantage
  ...
        ↓ concatenate in time order
  one rollout-wide normalization
```

对 agent \(i\)：

$$
\hat A_t^i =
\frac{A_t^i-\mu_i}{\sigma_i}
$$

\(\mu_i,\sigma_i\) 只来自该 agent 当前 rollout 的有效样本。不同 agent 不混合，不跨 update 累积。若样本少于 2 条或标准差不超过 \(10^{-6}\)，实现保留原始有限 advantage，避免除零或放大浮点噪声，并记录 `applied=0`。

## 4. Actor 与 Critic 的目标必须分离

Actor 的 PPO surrogate 使用标准化 advantage：

$$
L^{clip} =
\mathbb{E}\left[
\min\left(
\rho_t\hat A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right]
$$

Critic 的 return 必须使用原始 advantage：

$$
R_t^{raw}=A_t^{raw}+V_{old}(s_t,i)
$$

不能使用 `normalized_advantage + old_value`，否则 Critic 会脱离实际奖励尺度。生产代码对这一等式进行 `atol=1e-6` 的运行时校验。

## 5. PPO 更新前为什么必须满足 ratio=1

采样阶段保存 tanh 变换前的高斯变量 \(z\) 和旧 log-prob。更新阶段用同一个 \(z\) 重算：

$$
\rho_t =
\exp\left(
\log\pi_\theta(z_t|o_t)
-
\log\pi_{\theta_{old}}(z_t|o_t)
\right)
$$

在任何 Actor 更新之前，当前参数仍等于采样参数，因此应满足：

$$
\max_t |\rho_t-1| < 10^{-6}
$$

若不满足，通常说明 observation、动作、旧 log-prob、agent 或时间顺序错位。P0-00D 在第一个 Actor `optimizer.step()` 前按 agent 检查完整 rollout；测试同时验证篡改参数会触发 fail-fast。

## 6. 数值健康保护

每个 Actor/Critic minibatch 都检查：

- value prediction、return、new log-prob、entropy、ratio 和 loss 有限；
- approximate KL 有限且不出现异常负值；
- clip fraction 位于 `[0,1]`；
- backward 后所有 gradient 有限；
- `clip_grad_norm_` 使用 `error_if_nonfinite=True`；
- optimizer step 后所有参数仍有限。

Actor gradient norm 按全部 minibatch 汇总，不再只记录最后一个 minibatch。TensorBoard 同时记录每个 agent 的原始/标准化 advantage 均值和标准差、是否执行标准化，以及更新前最大 ratio 误差。

## 7. 与 P0-00B/P0-00C 的关系

```text
P0-00C active_mask
  决定 transition 是否存在
          ↓
P0-00B terminated/truncated
  决定边界与 bootstrap
          ↓
P0-00D GAE/PPO
  决定 Actor/Critic 如何使用有效样本
```

三者缺一不可：失活伪样本会污染 GAE；错误 T/X 会污染 bootstrap；错误标准化范围会改变 episode 间的相对更新权重。

## 8. 实施兼容边界

P0-00D 当前只修改冻结的基础策略入口 `train_no_rllib_mappo_robust_v2.py`。部分尚未进入当前阶段的对抗/PSRO 入口仍导入旧 `compute_gae_per_episode()`。为避免一次任务隐式改变多个训练协议：

- 新基础策略使用 `compute_raw_gae_per_episode()` 和 `normalize_advantages_over_rollout()`；
- 旧导入方暂由带明确说明的 compatibility wrapper 保持原行为；
- 对抗/PSRO 入口必须在各自阶段显式迁移并重新验收，不能把本次基础策略验收自动外推。

## 9. 五项验收结果（2026-07-23）

| 维度 | 结果 |
|---|---|
| GAE 边界正确性 | 手算双 episode 轨迹完全一致；改变后一个 episode 不影响前一个 |
| 标准化范围正确性 | 每个 agent 的 rollout 统一一次；非退化样本均值 0、标准差 1 |
| Actor/Critic 目标分离 | return 始终等于 `raw_advantage + old_value` |
| PPO 概率与数值健康 | 未更新策略 ratio 误差小于 `1e-6`；synthetic optimizer step 全部有限 |
| 集成与兼容性 | 新增 7 项与原 18 项共 25 项通过；兼容 wrapper 输出受测试保护；网络输入和 checkpoint 张量形状不变 |

## 10. 可用于论文的方法表述草稿

> 为避免多场景轨迹之间的回报估计相互污染，本文首先依据逐智能体 episode 边界独立计算未标准化的广义优势估计。真实终止切断价值 bootstrap，时间截断或同伴事件造成的外部截断则使用同帧最终状态价值，但二者均切断 GAE trace。随后，将同一智能体在当前 rollout 中的全部有效 advantage 按时间顺序拼接，并仅执行一次标准化，从而在稳定策略梯度尺度的同时保留不同 episode 之间的相对学习信号。
>
> 策略网络使用标准化 advantage 构造 PPO clipped objective，集中式价值网络则使用原始 advantage 与采样时价值之和作为回归目标。更新前通过保存的 pre-tanh 高斯变量重算动作对数概率，并要求未更新策略的概率比严格接近 1；训练期间进一步监控 KL 散度、裁剪比例、损失、梯度及参数有限性，以防止错误样本对齐或数值异常进入优化过程。

## 11. 参考资料

1. Schulman, J., et al. *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438, 2015. <https://arxiv.org/abs/1506.02438>
2. Schulman, J., et al. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017. <https://arxiv.org/abs/1707.06347>
3. Yu, C., et al. *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games*. NeurIPS Datasets and Benchmarks, 2022. <https://arxiv.org/abs/2103.01955>
