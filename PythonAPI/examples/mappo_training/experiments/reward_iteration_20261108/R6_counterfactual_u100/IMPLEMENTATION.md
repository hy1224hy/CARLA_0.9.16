# 实现索引

- source/environment/mut_mappo_outcome_reward_v6.py：在线成本白名单；删除任务通行和风险接近奖金，沿用物理诊断。
- source/evaluation/r6_counterfactual.py：固定时域红车结果积分、配对检查、逐车奖金、一次性结算、全样本分母和有符号效果汇总。
- source/training/r6_collection.py：三个顺序回合，原始基础策略替换，角色/步固定噪声；只有F进入PPO。
- source/training/train_mut_mappo.py：接入R6采样和counterfactual_episodes.jsonl；现有A/B完整边界保存、断点恢复、Actor/Critic更新保持原流程。
- source/evaluation/evaluate_r6_counterfactual.py：独立四模式评估，一个团队复用一个运行时；保存全部案例、配对效果和红车时间序列。
- source/training/mut_mappo_config.py：R6配置校验与源哈希；拒绝启用旧事件质量门、非B0或非CPU首版，要求继承42维团队Critic。
- config.json：Town05、seed42、100轮、每团队每轮16个真实回合；每个真实回合另有2个辅助回合；俯视观察follow_spectator=true；不改渲染质量。

source/是可独立运行的完整快照，附所需场景、可视化模块和CARLA Python agents依赖。无需覆盖项目根代码。vendor/agents来自本地CARLA包，仍使用原始agents导入名。

训练日志：episodes.jsonl只含真实PPO轨迹，counterfactual_episodes.jsonl明确标记辅助、ppo_samples=false；两者均含round/segment，恢复后按检查点round_segments过滤未完成/重复片段。metrics.jsonl.training_window.<team>.counterfactual含逐车合格数、拒绝原因、有符号效果、全样本奖金与奖励分量。完整tick账含三次轨迹。

评估日志：cases.jsonl含四模式全量回合；analysis.json保留逐场景边际、联合及交互效果；timelines/保留逐红车实际进度/负担/有效掩码，不伪造时域延长段。none模式结算R6训练奖金，其余模式的returns只有在线成本，不能直接用四模式总回报做效果比较，必须比较自然单位D/B与实际驾驶结果。

既有传感器回调时序未新增frame barrier，记录仍注明live_validation_pending；这会影响接触识别，需要现场核验。离线测试通过不是CARLA物理、传感器时序或对抗效果证据。
