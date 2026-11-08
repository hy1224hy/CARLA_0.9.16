# 使用方式

本轮仅编写代码，不启动R6，不抢占正在运行的R5a CARLA客户端。执行前确认当前训练/评估已释放2000端口所有权，CARLA由用户启动。不要用旧R5监督器启动R6，旧评估器不知道R6事后结算协议。

在本版本source/目录使用项目venv：

```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -B -X utf8 -m training.train_mut_mappo --config ../config.json --validate-only
```

独立R6训练命令（当前不执行）：

```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -B -X utf8 -u -m training.train_mut_mappo --config ../config.json --run-dir ../run --allow-candidate
```

完整边界恢复时追加`--resume`，不直接从旧R5a模型续训，本版Actor/Critic/ValueNorm均从原始基础检查点初始化。不能把旧版本奖励的回放样本混入R6。

训练结束后按顺序评估，不能与训练共占CARLA：

```powershell
& 'D:/study/yan2/CARLA_0.9.16/venv/Scripts/python.exe' -B -X utf8 -u -m evaluation.evaluate_r6_counterfactual --config ../config.json --checkpoint ../run/latest.pt --output-dir ../evaluation_selection --split selection
```

每团队24场景×4模式，A/B共192回合。selection反馈只能用于设计选择；参数冻结后用独立test，改`--split test --output-dir ../evaluation_test`。同一文件夹不覆盖重跑证据。评估必须同时保留合格与拒绝案例，base_u0为both替换模式。

正式论文前：校准场景/策略不变的重复模拟噪声，检查初始状态匹配拒绝率和传感器延迟，验证普通红车能力，确定效应门槛，多seed训练并报告计算量与不确定性。当前配置仍标记weak_red_diagnostic_only，不能作为正式鲁棒性结论。
