# staged_v1验证记录（2026-11-08）

## 最终源码检查

```bash
python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 highwaytest.tests.test_staged -v
```

47项通过，退出码0。包括原34项与新增13项：逐维探索边界、旧配置默认兼容、阶段晋降、坏参考拒绝晋级、阶段及模型精确续训、驾驶资格拒绝、训练seed演示、每车停车诊断、seed42矩阵、真实子进程共同初始化、整型/浮点型攻击权重预算，以及分发副本不污染源码身份。

日志：runs/staged_final_tests.log。服务器需在自身环境重新检查。

## 真实物理场景工程验证

runs/staged_releasecheck_s42：正常PPO预算1024，对抗预算仅32步（G4 smoke）；只使用gate和selection，没有运行真实最终test。

- IDM示范64回合，40轮拟合；BC在gate/selection每场景族均100%合格、0违规、0停车，较差蓝车平均进度约304–338m。
- 正常PPO1024步后，A/B在selection两场景族仍均100%合格、0违规；共同初始化导出再次通过gate/selection资格。
- G4从导出的共同模型开始32步对抗训练；2个selection工程案例100%合格、0违规，U约0.00295，较差进度330.61m。
- 课程仍为0.1，未进入完整攻击权重1；selected.eligible=false，符合阶段约束，不能作完整算法性能结果。
- GIF：runs/staged_releasecheck_s42/bootstrap_preview.gif，以及attack_smoke/G4_s42/evaluations/selection.gif。

物理闭环源码hash为d5b1d3375f44ad261c77ab23427d98f1f76fee2284423ba1ab61c42eceb6c017。最终发布相较它增加预算统计中整型/浮点权重等值处理及测试、共享初始化批量子进程测试，并修复源码指纹/快照排除dist旧副本及跨平台路径排序；初始化/奖励/采样/课程行为不变。最终源码重新完成全部47项测试。旧工程目录不可用最终源码原地resume；服务器新包从新study开始。

没有运行100k普通驾驶或8×200k长期对抗，没有证明长期稳定、算法优势或真实CARLA性能。包中不含本地checkpoint或演示数据，服务器会按固定流程重新生成并记录来源。
