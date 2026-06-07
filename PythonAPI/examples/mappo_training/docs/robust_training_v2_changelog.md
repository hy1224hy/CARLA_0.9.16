# train_no_rllib_mappo_robust_v2.py 改动记录

基于 `train_no_rllib_mappo_robust.py`（简称 v1）的改进版，针对 [rollout存在的问题.md](rollout存在的问题.md) 中记录的三大问题进行修复。

---

## 改动总览

| # | 问题 | 改动 | 影响 |
|---|------|------|------|
| 1 | Buffer 容量不足 | rollout_episodes 4→16, ppo_epochs 10→5 | 降低优势估计方差，减少数据过度复用 |
| 2 | GAE 跨 episode 泄漏 | 按 episode 边界独立计算 GAE | 修复截断 episode 导致的梯度污染 |
| 3 | 场景分布随机震荡 | 分层 + 加权采样 | 保证每个 update 的场景多样性 |
| 4 | 课程硬切换导致遗忘 | 阶段过渡期概率混合旧场景 | 平滑场景分布变化 |
| 5 | 无历史数据复用 | ~~可选混合 Buffer~~ 已移除 | 跨 update 混合采样与 PPO on-policy 不兼容，已删除 |
| 6 | Buffer 无边界信息 | 新增 episode_boundaries + bootstrap_states 追踪 | 支撑改动 2 和 GAE 截断 bootstrap |

---

## 改动 1：超参数调整

**问题**：v1 默认 `rollout_episodes=4`, `ppo_epochs=10`。4 个 episode 数据量通常仅几百到两千条 transition，却跑 10 轮 PPO epoch，数据被严重过度复用，优势估计方差极大。

**改动**：

| 参数 | v1 默认 | v2 默认 | 理由 |
|------|---------|---------|------|
| `--rollout-episodes` | 4 | 16 | 增大数据量 4×，降低方差 |
| `--ppo-epochs` | 10 | 5 | 数据量增大后可减少 epoch 避免过拟合 |
| `--min-transitions-per-update` | (无) | 512 | 极端小样本时跳过更新，防止噪声梯度 |

**代码位置**：`build_parser()` 中对应参数的 `default` 值，`train()` 中 `skip_update` 逻辑。

---

## 改动 2：修复 GAE 跨 episode 泄漏

**问题**：v1 将所有 episode 的 transition 拼接后一次性传入 `compute_gae()`。当 episode 因 `max_steps` 截断时，最后一帧 `done=False`，GAE 的 `mask=1`，导致优势估计跨越独立 episode 边界继续传播。例如：episode 1 截断于 step 800 (done=0)，episode 2 从 step 0 开始，但拼接后 GAE 在 step 799→800 时错误地引入了 episode 2 的信息。

**修复**：
1. `RolloutBuffer` 新增 `episode_boundaries: Dict[int, List[Tuple[int, int]]]` 和 `mark_episode_end(aid)` 方法
2. 每个 episode 结束时调用 `buffer.mark_episode_end(aid)`，记录该 episode 在当前 agent 数据中的 `(start, end)` 索引
3. 新增 `compute_gae_per_episode()` 函数，遍历边界列表，对每个 episode 独立计算 GAE 后拼接

**代码位置**：
- `RolloutBuffer.mark_episode_end()` 和 `episode_boundaries` 字段
- `compute_gae_per_episode()` 函数
- `train()` 中 episode 结束处调用 `buffer.mark_episode_end(aid)`
- PPO 更新处使用 `compute_gae_per_episode()` 替代 `compute_gae()`

---

## 改动 3：场景分层 + 加权采样

**问题**：v1 使用 `random.choice(available_scenarios)`，同一 update 的 4 个 episode 可能全是同一场景或完全随机，场景分布不可控。

**修复**：新增 `sample_scenarios_stratified()` 函数，支持三种模式：

| 模式 | 行为 |
|------|------|
| `random` | 纯随机（等同于 v1） |
| `stratified` | 每个可用场景至少 1 个 episode，剩余配额随机轮询 |
| `weighted`（默认） | 每个可用场景至少 1 个，剩余配额按失败率加权（困难场景上采样） |

新增参数 `--scenario-sampling-mode` 控制模式。

**代码位置**：
- `sample_scenarios_stratified()` 函数
- `train()` 中场景采样逻辑（调用该函数替代 `random.choice`）
- `build_parser()` 中 `--scenario-sampling-mode` 参数

---

## 改动 4：课程软过渡

**问题**：v1 的阶段切换是二元的：update < warmup_updates 时只有基础场景，≥ warmup_updates 后立即切换到下一阶段。导致策略梯度完全由新场景主导，旧场景经验全部丢失。

**修复**：新增 `--curriculum-transition-window` 参数（默认 10），在阶段切换点前后 `transition_window` 范围内线性过渡：

- `_compute_transition_alpha(update)` 返回 α ∈ [0, 1]
- `get_scenario_sampling_weights(update)` 返回新旧场景的混合权重
- α=0 时全用旧阶段场景，α=1 时全用新阶段场景，中间线性插值

**代码位置**：
- `CurriculumLearning._compute_transition_alpha()`
- `CurriculumLearning.get_scenario_sampling_weights()`
- `train()` 中调用 `get_scenario_sampling_weights()` 获取混合权重
- `build_parser()` 中 `--curriculum-transition-window` 参数

---

## 改动 5（已移除）：跨 update 混合 Buffer

**问题**：v1 是纯 on-policy PPO，每个 update 结束后数据立即丢弃。历史经验完全无法复用，加剧了场景切换时的遗忘。

**初版尝试**：新增 `MixedBuffer` 类和 `--use-mixed-buffer` flag，但存在两个问题：
1. 训练循环中初始化了 `MixedBuffer` 但从未调用 `add_batch()` / `sample()`，功能未实际接入
2. 即使接入，旧数据的 `log_prob` 来自旧策略，混入 PPO 更新会造成 off-policy 偏差，需要重要性采样权重修正

**最终处理**：**已从 v2 中移除**。`MixedBuffer` 类、`--use-mixed-buffer`、`--mixed-buffer-size` 参数均已删除。纯 on-policy PPO 下跨 update 经验复用需要更复杂的 IS ratio 修正，留作未来工作。

---

## 改动 6：RolloutBuffer 增强

**问题**：v1 的 `RolloutBuffer` 缺少 episode 边界信息，无法支持按 episode 独立计算 GAE。

**修复**：
- 新增 `episode_boundaries` 字段存储每个 agent 的 episode 起止索引
- 新增 `mark_episode_end(aid)` 方法供 episode 结束时调用
- 新增 `total_size()` 方法返回总 transition 数（用于诊断）

**代码位置**：`RolloutBuffer` 类的全部字段和方法。

---

## Bug 修复记录（v2 初版 Code Review 后修复）

### ✅ Bug 1：软过渡权重未传入采样函数

**问题**：`get_scenario_sampling_weights()` 返回 `{scenario: weight}` 字典，但调用处只取 `.keys()` 传入 `sample_scenarios_stratified()`，精心计算的过渡权重 α 完全丢失。软过渡退化成了"场景池扩展"。

**修复**：
- `sample_scenarios_stratified()` 新增 `external_weights` 参数
- weighted 模式下将 `external_weights` 与 `success_rates` 相乘得到最终采样权重
- 调用处传入完整 `soft_weights` 字典

### ✅ Bug 2：MixedBuffer 未接入训练循环

**问题**：`MixedBuffer` 类定义并初始化，但训练循环中无任何 `add_batch()` / `sample()` 调用。且旧数据的 `log_prob` 来自旧策略，混入 PPO 会造成 off-policy 偏差。

**修复**：删除 `MixedBuffer` 类、其初始化代码、及 `--use-mixed-buffer` / `--mixed-buffer-size` CLI 参数。

### ✅ Bug 3：截断 episode 的 GAE bootstrap

**问题**：`compute_gae()` 中 `next_value` 初始化为 `0.0`。对于 `max_steps` 截断的 episode（最后一帧 done=False），应该用 Critic 对终止状态的价值估计 V(s_T) 做 bootstrap，用 0 会低估最后几步的优势。

**修复**：
- `compute_gae()` 新增 `bootstrap_value` 参数（默认 0.0 向后兼容）
- `compute_gae_per_episode()` 新增 `bootstrap_values` 参数，与 boundaries 一一对应
- `RolloutBuffer` 新增 `episode_bootstrap_states` 和 `add_episode_bootstrap()` 方法
- 数据收集阶段：截断 episode 结束后立即捕获 `env.get_global_state()`
- PPO 更新阶段：用 Critic 计算截断 episode 的 V(last_global_state) 作为 bootstrap_value

### ✅ Bug 4：优势标准化作用域

**问题**：对所有 episode 合并后的优势做全局标准化 `(adv - mean) / std`，当 episode 间奖励量级差异大时（碰撞 -100 vs 完成 +200），全局标准化压缩了跨 episode 的相对差异。

**修复**：将优势标准化移入 `compute_gae_per_episode()` 内部，对每个 episode 独立标准化后拼接。

---

## 未改动

- 环境侧（`carla_multi_agent_env.py`）：场景加载、reset 逻辑不变
- 模型侧（`Actor`, `CentralizedCritic`）：网络结构不变
- v1 文件（`train_no_rllib_mappo_robust.py`）：原封不动，方便对比和回退
- 算法核心（MAPPO CTDE）：保持纯 on-policy PPO 架构

---

## 运行方式

```bash
cd PythonAPI/examples/mappo_training

# 默认参数运行（weighted 采样 + 软过渡）
python -m training.train_no_rllib_mappo_robust_v2 --render

# 对比 v1 行为（random 采样，无软过渡）
python -m training.train_no_rllib_mappo_robust_v2 \
    --rollout-episodes 4 --ppo-epochs 10 \
    --scenario-sampling-mode random \
    --curriculum-transition-window 0 \
    --render

# 完整参数示例
python -m training.train_no_rllib_mappo_robust_v2 \
    --updates 200 \
    --rollout-episodes 24 \
    --ppo-epochs 5 \
    --scenario-sampling-mode weighted \
    --curriculum-transition-window 15 \
    --use-mixed-buffer \
    --mixed-buffer-size 30000 \
    --render
```


完整读完了，现在逐项分析。

---

## 一、正确修复了原有问题

### ✅ GAE 跨 episode 泄漏（最重要的修复）

这是 v2 中**最关键也最正确**的修复。原版直接对整个 rollout 的拼接数据跑一遍 GAE，当多个 episode 连在一起时，第一个 episode 最后一步的 `next_value` 会用到第二个 episode 第一步的 value，造成优势估计被下一个 episode 的数据污染。

v2 通过 `mark_episode_end()` 记录边界，然后 `compute_gae_per_episode()` 按边界切片独立计算，逻辑是正确的。

但有一个**细节 bug**：`compute_gae()` 内部 `next_value` 初始化为 `0.0`，这对于因碰撞/到达终点而自然结束的 episode 是正确的（终止状态 bootstrap 为0），但对于因 `max_steps` 截断的 episode，最后一帧 `done=False`，此时应该用 `values[-1]` 或重新推断一次 Critic 来做 bootstrap，而不是直接用 0。截断 episode 用 0 会低估最后几步的优势，导致策略偏向保守。

### ✅ rollout_episodes 增大（4→16），ppo_epochs 减少（10→5）

方向正确。数据量变大后每次 PPO 更新的梯度估计更稳，同时减少 epoch 避免同一批数据过度复用导致的 off-policy 偏差。

### ✅ 场景分层 + 加权采样

`weighted` 模式下先保证每个场景至少出现一次，剩余配额按失败率加权，解决了原来纯随机可能导致某些场景长期未被采样的问题。逻辑正确。

---

## 二、新引入的问题（均已修复，保留原始分析供参考）

> **状态更新**：以下 4 个问题均已于后续两轮修改中修复。此处保留原始 code review 分析。

### ✅ 软过渡逻辑与实际采样脱节（已于第二轮修复）

软过渡的设计意图是在阶段切换时，通过 `get_scenario_sampling_weights()` 返回一个包含旧阶段场景的加权字典，让旧场景以衰减概率参与采样。但实际采样路径是这样的：

```python
# 软过渡权重计算
soft_weights = curriculum.get_scenario_sampling_weights(update)
weighted_scenarios = list(soft_weights.keys())   # 只取了场景名列表，权重丢了

# 然后用这个列表去调用分层采样
sampled_scenarios = sample_scenarios_stratified(
    available_scenarios=weighted_scenarios,       # 旧+新场景均在列表里
    n_episodes=args.rollout_episodes,
    success_rates=success_rates,                  # 成功率权重
    mode=args.scenario_sampling_mode,             # weighted模式
)
```

`soft_weights` 字典里精心计算的过渡权重（α值）完全没有被传入 `sample_scenarios_stratified()`，函数只接收了 `success_rates` 作为权重。结果是：过渡期内旧阶段场景和新阶段场景**被等概率对待**，软过渡退化成了硬切换+场景池扩展，和设计意图完全不符。

**修复**：`sample_scenarios_stratified()` 新增 `external_weights` 参数，调用处传入完整 `soft_weights` 字典。[详见第二轮修复记录](#-bug-1软过渡权重未传入采样函数)

### ✅ MixedBuffer 与 PPO 的兼容性问题（已于第二轮修复）

`MixedBuffer` 初始化了但在训练循环中**完全没有被使用**：

```python
mixed_buffer = MixedBuffer(...) if args.use_mixed_buffer else None
# 之后再也没有 mixed_buffer.add_batch() 或 mixed_buffer.sample() 的调用
```

即使启用 `--use-mixed-buffer`，跨 update 混合采样也不会发生，这个功能是个空架子。

退一步说，即使实现了，`MixedBuffer` 的设计也有根本缺陷：它存储的是 `Transition`，但里面的 `log_prob` 是旧策略产生的，混入后直接用于 PPO 更新时 ratio 可能偏离很远。虽然 clip 机制有一定过滤作用，但对于特别旧的数据（ratio 接近0或很大），clip 只是截断梯度而不能修正偏差，会导致 Actor 更新方向错误。正确做法是引入重要性采样权重，或者设置 ratio 阈值直接过滤掉过旧的样本。

**修复**：删除 `MixedBuffer` 类、初始化代码、及相关 CLI 参数。[详见第二轮修复记录](#-bug-2mixedbuffer-未接入训练循环)

### ✅ `ep_length` 计数问题（经核实无 bug）

`ep_length` 在循环开始就 `+1`，但 `ep_success` 的判断和 `break` 在循环末尾，所以实际步数计数是正确的。但注意 `ep_done_cleanly` 变量在 v1 中存在，v2 中已经删掉了，统计上没有区分"干净结束"和"超时截断"，只能从 `ep_length < max_steps` 来间接判断。

**结论**：非 bug，逻辑正确。

### ✅ 优势标准化的作用域问题（已于第二轮修复）

```python
adv_arr = (adv_arr - adv_arr.mean()) / (adv_arr.std() + 1e-8)
```

这里是对**单个 agent 所有 episode 合并后**的优势做全局标准化。对多 episode 合并的数据来说，如果各 episode 的奖励量级差异很大（比如一个 episode 碰撞了拿了 -100，另一个顺利完成拿了 +200），全局标准化会压缩掉跨 episode 的相对差异，使策略无法区分"这个 episode 整体做得好"和"这个 episode 整体做得差"。

**修复**：改为在 `compute_gae_per_episode()` 内部逐 episode 标准化。[详见第二轮修复记录](#-bug-4优势标准化作用域)

---

## 三、总结（三轮修改后最终状态）

| 改进项 | 状态 | 备注 |
|---|---|---|
| GAE 跨 episode 泄漏 | ✅ 已修复 | 按 episode 边界独立计算 + 截断 bootstrap |
| rollout 数据量扩大 | ✅ 有效 | 16 episodes 显著改善梯度方差 |
| 场景分层加权采样 | ✅ 有效 | 逻辑正确，external_weights 已正确传递 |
| 课程软过渡 (weight pass-through) | ✅ 已修复 | 第二轮：external_weights 传入采样函数 |
| 课程软过渡 (stage_start 对齐) | ✅ 已修复 | 第三轮：stage_start 公式与 _get_stage_idx 对齐 |
| 跨 update 混合 Buffer | 🗑️ 已移除 | 与 PPO on-policy 不兼容，已删除 |
| 优势标准化作用域 | ✅ 已修复 | 第二轮：改为逐 episode 独立标准化 |
| 逐 episode 标准化短 episode 保护 | ✅ 已修复 | 第三轮：std < 1e-6 时跳过标准化 |
| GAE 截断 episode bootstrap | ✅ 已修复 | Critic bootstrap 替代 0 值 |
| bootstrap Critic 一致性 | ✅ 无问题 | 第三轮核实：bootstrap 在 epoch 前计算，与数据收集同一 Critic |
| ep_length 截断判断 | ✅ 无问题 | 第三轮核实：逻辑正确 |


# 第三次修改

## 残留问题与新引入的问题（逐项核实 + 修复）

### ✅ Bug 1（严重）：软过渡 `stage_start` 边界计算偏差 [已修复]

`_compute_transition_alpha` 第 331 行原公式：

```python
stage_start = warmup + (full - warmup) * (stage_idx - 1) / max(n_stages - 1, 1)
```

与 `_get_stage_idx` 的过渡判定不一致。`_get_stage_idx` 用 `int(progress * n)` 判定：
- 阶段 0→1：`progress ≥ 1/3` → update ≈ 20（默认参数 warmup=10, full=40, n=3）
- 阶段 1→2：`update ≥ 40`（显式 `>= full_scenario_updates` 检查）

但 `_compute_transition_alpha` 算出的 `stage_start`：
- stage_idx=1: `10 + 30 * 0 / 2 = 10` → α 在 update 15 就到达 1.0，实际 stage_idx 在 update 20 才切换
- stage_idx=2: `10 + 30 * 1 / 2 = 25` → α 在 update 30 就到达 1.0，实际 stage_idx 在 update 40 才切换

**α 恒为 1.0，软过渡从未生效。** 修复：使 `stage_start` 与 `_get_stage_idx` 的实际过渡点对齐：

```python
if stage_idx == n_stages - 1:
    stage_start = float(self.full_scenario_updates)  # 末阶段：update >= full
else:
    stage_start = warmup + (full - warmup) * stage_idx / n_stages  # 中间阶段：int(progress*n) >= stage_idx
```

修复后（默认参数 n=3, warmup=10, full=40）：
- stage_idx=1: `stage_start = 10 + 30 * 1/3 = 20` ✓
- stage_idx=2: `stage_start = 40` ✓

过渡窗口（transition_window=10）在正确时机生效：阶段 0→1 在 update 15~25 软过渡。

### ✅ Bug 2（次要）：逐 episode 标准化缺少短 episode 保护 [已修复]

`compute_gae_per_episode` 第 181 行原代码：

```python
adv = (adv - adv.mean()) / (adv.std() + 1e-8)
```

碰撞 episode（3~5 步）的 advantage 值几乎相同，`std()` 接近 0，标准化会放大噪声。修复：添加 std 下限保护：

```python
ep_std = adv.std()
if ep_std > 1e-6:
    adv = (adv - adv.mean()) / (ep_std + 1e-8)
```

### ❌ bootstrap 的 Critic 与数据收集时的 Critic 不一致 [不成立]

bootstrap_value 在 PPO 更新块内、epoch 循环**之前**计算（第 639-645 行），此时 Critic 尚未经过任何梯度更新，与数据收集时（第 529 行）是同一个预更新 Critic。`V_new` 和 `V_old` 实际上是同一个 Critic，不存在偏差。此问题标记为误判。

### ❌ `ep_length` 的截断判断有差一错误 [不成立]

`ep_length` 从 0 开始，循环内第一行 +1，跑满时最终 = 1000，`>= max_steps(1000)` 为 True；提前结束时 < 1000 为 False。逻辑正确，此问题标记为无误。

2026/5/28 改动摘要
改动摘要

models/actor_critic.py

A3：Actor.__init__ 新增 init_log_std=-0.5（std≈0.6，旧默认 1.0 易让 tanh 输出饱和，是 approx_kl 偏大的元凶之一）
training/train_no_rllib_mappo_robust_v2.py

A1：ep_success 改为严格 "所有 agent 到 reached_target"；ep_collision / ep_timeout 单独分类；TB 新增 episode/rate_success_to_goal / rate_collision / rate_timeout 三条独立曲线 + cumulative 版本。修掉了"超时被算成成功"污染课程学习的核心 bug
A2：把"自然终止" per-agent 化——只有自己撞 / 自己到点才 bootstrap=None；其余（max_steps 截断 + 被对方 severe collision 牵连）一律用 V(last_global_state) bootstrap，修掉 v2 changelog 只覆盖 max_steps 那一半的遗漏
A3：
新增 --target-kl（默认 0.02）、--lr-decay/--no-lr-decay（默认开）、--init-log-std（默认 -0.5），--clip-eps 默认 0.2→0.15
lr 每 update 起始线性衰减到 5%
actor PPO 内 epoch 计算 Joschu 风格 approx_kl = (r-1) - log r，超过 1.5 × target-kl 则该 agent 跳过后续 epoch；TB 新增 ppo/stop_epoch_agentX 诊断
A4：每 update 维护 best.pt（按窗口严格 success_rate 取 max），评估时直接拿它
训练日志 print 改为 success / collision / timeout 三段并显示当前 lr_a；TB 加 train/lr_actor、train/lr_critic、train/log_std_agentX
验收：跑 30 update sanity，TB 上看

rate_success_to_goal + rate_collision + rate_timeout ≈ 1.0（三类终止覆盖所有 episode）
ppo/approx_kl_agentX < 0.03，clip_frac < 0.3，stop_epoch < ppo_epochs 偶尔触发（说明早停在工作）
rate_success_to_goal 大致单调上升而不再剧烈震荡

