# 启动与验证

本版等待用户启动CARLA。不得把旧u100检查点传给--resume；第一次必须从本版u0开始。当前目录尚无run，证明未启动。

工作目录：D:/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training

配置检查（不连接CARLA）：
```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -B -m training.train_mut_mappo --config experiments/reward_iteration_20261108/R5a_teamcritic42_u100/config.json --validate-only
```

用户启动后，由当前聊天先确认无重复worker再启动训练；基础模块命令如下，不会在准备配置时自动执行：
```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -X utf8 -u -m training.train_mut_mappo --config experiments/reward_iteration_20261108/R5a_teamcritic42_u100/config.json --run-dir experiments/reward_iteration_20261108/R5a_teamcritic42_u100/run --allow-candidate
```

训练后再按原协议顺序执行base_u0/候选selection及A/B四模式替换评估。不要重复启动旧R5a失败监督器。启动前应把周期监控指向本版；当前control/state.json仍如实保留旧版失败状态，不伪装成本版运行中。

VERIFICATION.json为真实基础权重、初始预测、独立参数与未改奖励的离线核验；不代表实时仿真训练已通过。
