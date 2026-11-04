# MutMAPPO 的学习信号与 TensorBoard

`training.train_mut_mappo` 和 `utils.run_first_point_experiments train/run` 默认开启 TensorBoard，不需要改变奖励或实验配置。仅记录完整训练边界：先完成所有团队更新并提交 checkpoint，再写 event。横轴是 `actual_counters.train`（A+B 合计，包含训练 setup；不含 gate 验证）。单团队 MAPPO/IPPO 也适用。

## 新训练实时查看

沿用原训练命令即可。启动时输出 `[TensorBoard] .../RUN_DIR/tb`。默认目录为各 run 的 `tb/A`、`tb/B`，因此 TensorBoard 指向实验 `runs` 能同时比较各组/团队。

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training

# EXP 指向已经准备好的新实验目录；不要指向已训练完的旧实验要求其跨版本续训。
python -u -m utils.run_first_point_experiments train \
  --experiment "$EXP" --groups B0 B1 B2 --seeds 42 \
  --monitor-mode web --monitor-port 8080
```

另开服务器终端或 screen 窗口：

```bash
conda activate carla-mappo
tensorboard --logdir "$EXP/runs" --host 127.0.0.1 --port 6006
```

本地 SSH 转发（替换 SSH_ALIAS 为日常服务器连接别名，保留自己的 SSH 端口配置）：

```bash
ssh -N -L 16006:127.0.0.1:6006 SSH_ALIAS
```

本地浏览器打开 `http://127.0.0.1:16006/`。俯视画面仍使用已有 8080 转发；两者独立。首个完整训练边界完成后才有新数据，不是每个 simulator tick 刷新。

`--no-tensorboard` 可关闭 event 输出、保留 JSONL。`--tensorboard-dir PATH` 可改输出：直接训练写 PATH/A、PATH/B；实验 runner 写 PATH/RUN_ID/A、PATH/RUN_ID/B。不要让不同实验共用同名输出目录。TensorBoard 缺失会在连接 CARLA 前提示安装或关闭；运行中 event 写入失败会警告并关闭 TensorBoard，训练及 JSONL 继续。

## 已完成实验导出（不重训、不连接 CARLA）

```bash
python -m utils.mut_mappo_tensorboard \
  --experiment "$EXP" --output "$EXP/tensorboard_export"

tensorboard --logdir "$EXP/tensorboard_export" --host 127.0.0.1 --port 6006
```

也支持 `--run-dir /path/to/single_run --output /path/to/output`。导出只读取 metrics.jsonl 和 manifest，不导入训练环境或加载 checkpoint。输出每 run/每团队分别保存。历史日志没有的 advantage、学习率等字段不会伪造；没有 TTC 的样本不会填零。完整 JSONL 损坏会报错，未完成的尾行不导出，超出 manifest 安全边界的记录也不导出。

重复导出/同代码版本恢复训练，会利用 SummaryWriter 的 purge 语义重建 event 视图，再从权威 JSONL 回放；不删除旧 event 文件，也不修改模型/原始记录。输出目录中的 projection_source.json 绑定原 run 的绝对路径及身份摘要，防止同名 B0_s42 等不同实验互相覆盖显示；不同来源会明确拒绝并要求换输出目录。不要同时向同一 event 目录运行两个导出/训练写入进程。

## 优先看哪些图

每个 run 的 A/B 是相互依赖的候选团队，不能当作独立随机种子。`training_window/*` 是训练窗口统计，不能冒充独立测试。

| 标签 | 含义 |
|---|---|
| `training_window/reward_components_mean/*` | 每回合累计奖励分量的窗口均值 |
| `training_window/reward_components_per_decision_step/*` | 分量累计总和 / 该窗口决策步总和，防止回合长短混淆 |
| `training_window/reward_components_nonzero_episode_rate/*` | 有该奖励信号的回合比例，不是时间步密度 |
| `training_window/blue_eligible_rate` | 双蓝车均未触发碰撞、越界、道路规则或任务违规的比例；不等于双方到达率或全面舒适稳定 |
| `training_window/blue_collision_episode_rate` 等 | 四类硬约束分别发生的比例，可重叠；碰撞不含责任归因 |
| `training_window/safe_critical_event` | 最终硬约束合格且发生危险事件的比例；仍应结合 moderate/severe 的独立评估 |
| `training_window/primary_bonus_finally_ineligible_rate` | 获得主事件奖励的回合中，最终不合格的条件比例；同时记录分子和分母 |
| `training_window/critical_min_ttc_s_mean_observed` | 仅有 TTC 观测的回合，其最小 TTC 的均值；同时查看 observed_episodes |
| `ppo/advantage_stats/2/raw_mean`、`raw_std` 等 | 蓝车角色原始/归一化优势统计；角色3同理 |
| `ppo/learning_rates/2/actor/item_1` | optimizer 第一个参数组的真实学习率；不是根据 tick 估算 |
| `ppo/actor_loss_2/mean`、`surrogate_loss_2/mean` | epoch 列表的算术均值；`item_1`、`item_2` 是对应 epoch，不是训练时间 |
| `ppo/critic_loss/mean`、`explained_variance`、`value_error_abs_mean` | 归一化 Huber loss，以及原尺度价值估计诊断 |
| `ppo/approx_kl_2/mean`、`clip_frac_2/mean`、`post_tanh_entropy_2/mean` | 单次 PPO 新旧策略变化、裁剪比例和动作分布熵 |
| `ppo/grad_norm_actor_2/mean`、`grad_norm_critic/mean` | 裁剪前梯度范数；列表元素是 minibatch 诊断，不能解释成 epoch |
| `ppo/mutual_gated_kl_2`、`gate_nonzero_coverage_2` | 实际获准参与教学的差异与覆盖率；beta 非零不等于教学生效 |
| `gate/*` | gate 版本、各场景家族的资格/质量统计及放行标志 |

数值列表统一显示 `/mean`、`/max` 和 `/item_N`；列表位置只作索引诊断。每个字段原有的 epoch/minibatch/parameter-group 语义保持不变。缺失或非有限数据跳过，不用零代替。

## 旧 checkpoint 的兼容性

本项目将训练源码 SHA 纳入 run identity。本次 train_mut_mappo/mut_mappo_learning 的日志补充也会改变源码身份，因此旧实验不能用更新代码直接 `--resume`，原有严格校验保持不变。已完成旧实验只需导出；仍需完成的旧训练可保留原代码完成后导出。跨代码版本训练恢复需要显式迁移/分叉并保存来源，不能修改旧 manifest 绕过身份检查。

此改动不改变 R2 奖励、资格阈值、PPO 算法或学习率调度。新增优势统计是底层已经计算的 prepared.advantage_stats，真实学习率直接读取 optimizer；不会重新归一化或改变训练张量。

SummaryWriter 的写入、flush 和 purge 语义参见 [PyTorch 2.9 官方文档](https://docs.pytorch.org/docs/2.9/tensorboard.html)。
