
# 项目文件功能索引

本文档列出项目中所有文件的功能说明，帮助快速定位需要的文件。

> **当前入口冻结（2026-07-20）**：基础策略只使用 `train_no_rllib_mappo_robust_v2.py`，Phase 1 单蓝车只使用 `train_2red_1blue_ppo.py`。RLlib、shared-policy、minimal/robust v1、旧 adversarial PPO 和旧 multi 入口仅保留作历史参考，不得用于新实验。执行顺序见 [毕业课题研究推进 TODO](docs/毕业课题_研究推进TODO.md)。

---

## 目录

1. [环境模块](#环境模块)
2. [训练脚本](#训练脚本)
3. [评估脚本](#评估脚本)
4. [可视化模块](#可视化模块)
5. [工具模块](#工具模块)
6. [数据与配置](#数据与配置)
7. [文档](#文档)

---

## 环境模块

| 文件名 | 功能说明 | 关键类/函数 |
|--------|----------|-------------|
| **carla_multi_agent_env.py** | CARLA多智能体环境（核心文件） | `CarlaMultiAgentEnv` |
| | - 连接CARLA服务器、生成车辆 | `__init__()` |
| | - 观察空间：19维（8基础 + 11雷达） | `_get_observations()` |
| | - 动作空间：3维（throttle, steer, brake） | `step()` |
| | - 碰撞/车道偏离检测 | `_attach_collision_sensor()`, `_attach_lane_invasion_sensor()` |
| | - 正常智能体奖励函数 | `_compute_rewards()` |
| | - 对抗智能体专用奖励 | `_compute_adversarial_reward()` |
| | - 自定义起始位置 | `set_custom_start_positions()` |
| **carla_rllib_wrapper.py** | **[DEPRECATED]** 历史 RLlib 环境包装器 | `RLlibCarlaEnv` |
| **speedup_env_wrapper.py** | 提速环境包装器（用于评估） | `SpeedupEnvWrapper`, `SpeedupActionWrapper`, `AdaptiveSpeedupWrapper` |
| | - 固定油门放大模式 | `SpeedupActionWrapper` |
| | - 自适应速度调整模式 | `AdaptiveSpeedupWrapper` |

---

## 训练脚本

| 文件名 | 功能说明 | 使用场景 |
|--------|----------|----------|
| **train_no_rllib_mappo_robust_v2.py** | **[CURRENT]** 基础红车 MAPPO 唯一入口 | P0-00 修复/smoke，门槛通过后正式基础策略训练 |
| **train_2red_1blue_ppo.py** | **[CURRENT]** 单蓝车 PPO 唯一入口 | P0-06 通过后的 Phase 1 |
| **lifecycle_semantics.py** | **[CURRENT]** 统一校验 `done/terminated/truncated` 并选择合法 bootstrap state | 两个当前入口共享的 P0-00B 协议 |
| **checkpoint_protocol.py** | **[CURRENT]** 版本化 checkpoint、原子保存/SHA256、manifest、RNG 捕获恢复和严格兼容校验 | `robust_v2` 的 P0-00F 协议；无 CARLA 单测入口 |
| **metrics_protocol.py** | **[CURRENT]** 版本化奖励分量/raw diagnostic schema、有限值和奖励守恒校验 | `robust_v2` 的 P0-00F-TB 指标协议；不依赖 CARLA/TensorBoard |
| **experiment_contract.py** | **[CURRENT]** P0-BASE 严格 JSON 契约、正式 CLI 约束、关键源码预检、resolved config/manifest/registry 原子证据链 | P0-BASE-01；在 CARLA/optimizer 前 fail-fast |
| **train_rllib.py** | **[DEPRECATED]** 历史 RLlib 共享 PPO | 禁止新训练/论文结果 |
| **train_no_rllib_mappo.py** | **[DEPRECATED]** 早期独立 MAPPO | 历史参考 |
| **train_no_rllib_mappo_robust.py** | **[DEPRECATED]** robust v1 | 历史参考 |
| **train_no_rlib_sharedpolicy.py** | **[DEPRECATED]** 早期共享策略 PPO | 算法数据流未迁移，禁止使用 |
| **train_adversarial_ppo.py** | **[DEPRECATED]** 早期单对抗 PPO | 历史参考 |
| **run_adversarial_training_with_viz.py** | **[DEPRECATED]** 早期对抗训练启动器 | 历史参考 |
| **train_multi_adversarial_*.py** | **[LATER]** 多蓝车训练入口 | Phase 3 前统一生命周期/奖励后重新验收 |
| **pretrain_with_manual_data.py** | 手动驾驶数据预训练 | 用行为克隆初始化策略 |

---

## 评估脚本

| 文件名 | 功能说明 |
|--------|----------|
| **base_c0_protocol.py** | **[CURRENT]** 无 CARLA 依赖的 C0 严格契约、checkpoint provenance、case 展开、逐回合 schema、汇总和重复性协议 |
| **evaluate_base_policy_c0.py** | **[CURRENT]** 两红车确定性 C0 执行器；按 active-mask 运行 smoke/gate/held-out validation，支持精确前缀恢复 |
| **validate_base_c0_run.py** | **[CURRENT]** 独立重算 JSON/CSV、校验 SHA/case/清理并可比较双 Run 离散重复性 |
| **evaluate_rllib.py** | **[DEPRECATED]** 评估历史 RLlib 智能体 |
| **evaluate_rllib_checkpoint.py** | **[DEPRECATED]** 加载历史 RLlib checkpoint |
| **evaluate_no_rllib_mappo.py** | 诊断历史独立 MAPPO checkpoint；非当前统一 C0/C1/C2 结论入口 |
| **evaluate_adversarial_ppo.py** | 诊断历史对抗 checkpoint；非当前统一 C0/C1/C2 结论入口 |
| **evaluate_adversarial.py** | 对抗场景综合评估 |
| **evaluate_with_speedup.py** | 在提速场景下评估（测试对抗智能体鲁棒性） |
| **smoke_test_success_logic.py** | **[CURRENT]** 不加载 checkpoint，以顺序传送和近终点固定油门在真实 CARLA 中验收 success 奖励与生命周期；每次写 `result.json`/`steps.jsonl` |

---

## 可视化模块

| 文件名 | 功能说明 |
|--------|----------|
| **visualize_adversarial_training.py** | 对抗训练实时可视化 |
| | - pygame窗口显示CARLA俯视画面 |
| | - 实时显示各智能体奖励曲线 |
| **mappo_visualizer.py** | MAPPO训练可视化工具 |

---

## 工具模块

| 文件名 | 功能说明 |
|--------|----------|
| **training_metrics_recorder.py** | **[CURRENT]** 按 active agent 聚合奖励/PPO，并写入 TB、逐 update metrics 和逐 episode 生命周期记录 | 当前 `robust_v2` 的模块化指标入口 |
| **validate_tensorboard_run.py** | **[CURRENT]** 离线校验指标 schema、逐 episode 对账、PPO 门槛、update 顺序及 TensorBoard/JSONL 一致性 | P0-00G smoke 和后续正式 Run 的验收命令 |
| **mappo_logger.py** | **[LEGACY]** 旧训练日志工具；未实现当前固定 schema、active ownership 和 checkpoint 重放 | 历史入口参考，不用于当前 `robust_v2` |
| **manual_data_loader.py** | 手动驾驶数据加载器 |
| | - 加载、解析手动采集的数据 |
| | - 数据过滤、统计 |
| | - 转换为RLlib格式 |
| **record_manual_control.py** | 手动驾驶数据采集工具 |
| | - pygame控制界面 |
| | - 记录观察、动作、奖励轨迹 |

---

## 数据与配置

| 文件名 | 说明 |
|--------|------|
| **data/custom_positions_example.json** | 自定义车辆起始位置示例 |
| **data/manual_driving_data/** | 手动驾驶数据存储目录 |
| **models/mappo_models/** | MAPPO模型定义目录 |
| **logs/<RunID>/latest.pt** | 每个完整 update 原子替换的严格 resume checkpoint |
| **logs/<RunID>/periodic/** | 不可覆盖的周期 checkpoint 与 SHA256 sidecar |
| **logs/<RunID>/checkpoint_manifest.json** | Run checkpoint 路径、update 与 SHA256 登记 |
| **logs/<RunID>/tb/** | 当前 Run 的 TensorBoard event；与 JSONL 使用同一 update payload |
| **logs/<RunID>/metrics.jsonl** | 可审计的逐 update scalar payload；严格递增并可由 checkpoint 重放 |
| **logs/<RunID>/episodes.jsonl** | 可审计的逐 episode 终局与 per-agent active steps/transition/T-X/bootstrap 记录；resume 裁剪安全边界后的记录 |
| **logs/<RunID>/metrics_schema.json** | 指标协议、奖励字段、agent 和聚合口径 |
| **logs/<RunID>/resolved_config.json** | P0-BASE 正式 Run 的解析参数、契约/Git/源码/场景指纹与停止规则 |
| **logs/<RunID>/experiment_manifest.json** | P0-BASE Run 的 PREPARED/RUNNING/SEGMENT_COMPLETED/INTERRUPTED/ERROR 状态 |
| **configs/base_mappo/p0_base_pilot_v1.json** | seed 42 基础 MAPPO pilot 的机器可读正式参数真值与登记停点 |
| **configs/evaluation/p0_base_c0_eval_v1.json** | 基础 C0 的环境、checkpoint 资格、smoke/gate/validation cases、终局和输出机器真值 |
| **data/evaluation_scenarios/base_c0_validation_v1/** | 与训练场景按内容 SHA 隔离的 8 个 Town02 held-out 基础策略验证场景 |
| **experiments/base_mappo_registry.csv** | 每个正式基础 Run 一行的跨 Run 机器可读索引 |
| **evaluations/<EvalID>/** | C0 resolved config、manifest、逐 episode JSONL、summary JSON/CSV 与 validator report |
| **requirements.txt** | Python依赖包列表 |

---

## 文档

| 文件名 | 说明 |
|--------|------|
| **README.md** | 项目主文档（快速开始、安装、基本使用） |
| **FILE_INDEX.md** | 本文档：文件功能索引 |
| **MANUAL_DATA_USAGE.md** | 手动驾驶数据采集与使用详细指南 |
| **docs/MAPPO_checkpoint与恢复训练协议.md** | P0-00F fresh/pretrained/resume 语义、目录结构、兼容规则和标准命令 |
| **docs/MAPPO_TensorBoard指标协议.md** | P0-00F-TB 奖励 schema、聚合口径、PPO/模型指标、事务顺序和验证命令 |
| **docs/P0-00G_真实CARLA_smoke运行规范.md** | P0-00G 环境门槛、冻结命令、受控中断、预期产物与 PASS/HOLD 标准 |
| **docs/P0-00G_真实CARLA_smoke测试记录.md** | P0-00G 每次环境、fresh、中断、resume、validator、失败根因、修复证据与 Judge 裁决 |
| **docs/P0-00G-SUCCESS_真实success逻辑测试规范与记录.md** | 无基础 checkpoint 的 success/reward/lifecycle 两层真实 CARLA 测试、PASS 门槛、命令和逐次结果 |
| **docs/P0-BASE-01_基础策略正式实验契约.md** | 正式参数依据、命令、证据链、停止规则、风险边界和验收 |
| **docs/P0-BASE-02_基础策略C0评估协议.md** | 基础策略 C0 定义、三个 profile、checkpoint 资格、证据 schema、运行/复核命令和风险边界 |
| **docs/P0-BASE_基础策略实验记录.md** | fresh/resume/C0/GO-HOLD 逐 Run 人工实验台账模板 |
| **tests/test_checkpoint_resume.py** | 原子保存、哈希、RNG/optimizer 连续、协议拒绝和训练入口 checkpoint 回归测试 |
| **tests/test_training_metrics.py** | 奖励守恒、字段拒绝、ragged agent/稀疏事件聚合、PPO/模型指标、TB/JSONL 对账及 checkpoint 重放测试 |
| **tests/test_success_logic_smoke.py** | 近终点场景几何与 success reward/terminated/truncated/active/global-reason 审计器回归测试 |
| **tests/test_experiment_contract.py** | JSON schema、全批上界、CLI 覆盖拒绝、身份漂移和 registry/manifest 原子更新测试 |
| **tests/test_base_c0_evaluation.py** | C0 场景隔离、checkpoint provenance、确定性 active-agent runner、奖励守恒、恢复顺序、summary 防篡改和重复性测试 |

---

## 工作流程参考

### 正常智能体训练流程
1. `record_manual_control.py` → 采集手动数据（可选）
2. `pretrain_with_manual_data.py` → 预训练（可选）
3. 完成 TODO P0-00 和 Phase 0 门槛
4. `train_no_rllib_mappo_robust_v2.py` → 唯一正式基础策略入口
5. 独立 C0/C1 验证 → 生成 `best_base_valid.pt`

### 对抗智能体训练流程
1. （前置）P0-06 通过并取得 `best_base_valid.pt`
2. `train_2red_1blue_ppo.py --pretrained-ckpt <path>` → Phase 1 单蓝车训练
3. C0/C1/C2 统一评估器 → 配对独立评估与 `best_valid.pt`
4. Phase 1 通过后才开放多蓝车、PSRO 和通信实验

### 文件依赖关系
```
train_2red_1blue_ppo.py
├── carla_multi_agent_env.py
├── actor_critic.py
├── adversarial_reward.py
└── mappo_visualizer.py (可选)

evaluate_with_speedup.py
├── carla_multi_agent_env.py
└── speedup_env_wrapper.py

pretrain_with_manual_data.py
├── manual_data_loader.py
└── carla_rllib_wrapper.py (可选)
```
