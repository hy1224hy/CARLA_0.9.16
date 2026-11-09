# 渲染异常与地图重载修正

用户提供的截图为“Out of video memory trying to allocate a rendering resource”。R2 gate在121/144回合后客户端退出码0xC0000409；实际进程核验没有存活CARLA/评估控制器。R2蓝车训练已完成43轮，固定checkpoint SHA仍为4fcc1a7bce140871f14ce709fcbc4edd27302dafcc64562d2775e57a66c1108a，不重新训练、不更改已完成配置。

已核实：机器为RTX4060 Laptop、8188MiB；崩溃后nvidia-smi仅记录87MiB使用量，这不是崩溃前峰值，不能据此证明显存泄漏。保存的GameUserSettings.ini各画质档位为3。PPO配置为device=cpu，车辆RGB相机关闭，但服务器viewport仍在渲染。普通训练/参考复用一个runtime；原独立评估每回合新建runtime并无条件load_world(Town05)，造成反复地图加载，是重点待验证因素。

旧CrashContext中2026-11-01 12:39记录的是GameThread等待RenderThread超过120秒，不是这次截图的完整崩溃记录；其中Intel主显示适配器字段也不能单独证明CARLA实际使用了核显。不要把旧日志直接当成本次显存错误根因。

已修改实际utils.evaluate_mut_mappo_critical入口：默认复用已加载的同名空闲地图，换地图才load_world；检查车辆/传感器/行人/控制器残留，不满足条件就拒绝。仍保留逐回合全新runtime与正常reset/cleanup，CARLA所有权不变，不并行tick。覆盖仅在该评估作用域生效，异常时恢复原连接器；--no-reuse-idle-world保留旧行为用于复现。奖励、训练源码、配置、红方和检查点不变。

新评估在operation.json与evaluator_source.py记录运行方式、源码SHA及副本，每5秒记录只读整卡GPU显存（不是CARLA单进程归因）。43项相关离线检查通过，包括同图不重载、异图加载、占用拒绝和异常恢复；固定R2 checkpoint严格离线预检通过。这些不等于真实显存稳定性已经验证。

服务器重启建议显式使用-quality-level=Low，保留俯视。没有自动启动CARLA或重试；待用户服务就绪后，以新attempt完整配对评估，保留121回合失败目录及成本下界。恢复命令在mappo_training工作目录使用项目venv：-m utils.recover_critical_reward_evaluation --root experiments/adversarial_u1000_critical_reward_20261101_v4 --group R2 --attempt 1 --reuse-idle-world。正常退出与配对完整性仍须核对。

PyTorch/CUDA能力独立检查见torch_cuda_capability.json。PyTorch框架与CPU/CUDA设备是两回事；本轮沿用的cpu配置并不意味着没有使用PyTorch。后续若变更训练设备，应新建配置并登记硬件变量，不能改写本轮配置或将它冒充相同实验。
