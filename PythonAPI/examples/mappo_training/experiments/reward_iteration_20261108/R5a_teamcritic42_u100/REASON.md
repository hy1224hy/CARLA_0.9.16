# R5a critic修复对照

用户要求保留每团队两车MAPPO和独立critic，继承基础critic；A/B仅在启用互学习时通过actor辅助损失联系。本次仍为B0，不启用互学习。

对照为R5a_outcome10_u100。历史结果保留；原版critic实际输入是两红两蓝观测80维+4维身份，不是合并A/B两组蓝车。新critic只看本组蓝车，40维观测+2维局部身份。四车环境及80维回放存储格式保留，适配器在进入MLP前丢弃红车槽位，物理ID2/3映射为局部0/1。红车actor继续冻结。

初始化：从同一已核验基础u001000.pt严格加载两蓝车actor、critic以及配套ValueNorm；不继承旧优化器、基础训练计数或R5a u100参数。A/B权重初值相同，参数内存及优化器独立。ValueNorm输出原为归一化价值，因此必须配套复制统计以保持初始反归一化预测一致；统计计数1000是基础统计历史，不是本版训练进度。critic/actor随后用新对抗奖励同时更新，不增加critic预热。

不改奖励：outcome_weight=10、delay_fraction=.7、budget=8、progress=.03、control=.3、collision_scale=1；其他task、参考库、PPO设置及学习率计划全部与R5a一致。Town05，B0，seed42，每版100 updates，保留渲染。

这是critic观测范围+初始化+配套归一化的修复组合，不是仅奖励变化，也不是分别隔离三者的因果消融。不能保证收敛；需比较EV、早期价值偏差、总回报与逐车对抗分量，并最终用base_u0和非接触策略替换评估判断效果。旧R5a四模式评估失败仍保留原状态，不宣称补齐。

新旧critic配置与model_spec不同，禁止把旧84输入检查点续训到本版。旧配置仍可构造旧critic；新配置需显式声明blue_team/base_checkpoint，缺失或不兼容权重直接报错，不能悄悄退回随机初始化。

修复前代码提交：44b1e53。代码已准备，尚未连接CARLA或开始本版训练。

首次启动在u0保存前被旧checkpoint validator的critic_id_dim=4硬编码拒绝，未采样、未保存检查点。已显式按critic作用域验证2/4维身份及42维权重；14项相关回归含真实保存读取通过。失败记录保留在failed_start_u0_20261108_1648，从基础u0重新启动，奖励/初始化方案不变。
