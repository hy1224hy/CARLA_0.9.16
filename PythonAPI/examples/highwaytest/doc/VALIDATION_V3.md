# v3实现验证记录

实现目标：公平检验算法设计与优势，并支持在CARLA复验该结论；不保证优势存在，不将接口通过当成跨环境性能证据。

## 已完成

- 原有18项与新协议16项，共34项检查全部通过。
- 正常交通校准：20s时窗、两场景族各8例，允许红车换道；C0/C1有效率100%，C2同为正常IDM时16/16安全完成，额外攻击效用为0。
- 检查提前失败被标记censored、原始距离差置null、攻击效用为0；未完成目标的停车样本也不给有效攻击分。
- 折扣攻击势函数在安全失败或未完成目标时返还，测试折扣和为0，防止反复刷增量。
- 到达目标不会提前终止固定时间窗；时钟进入actor/critic；非整数策略时窗被拒绝。
- 局部特征经过世界平移/旋转后保持一致；CARLA路线投影、角度/速度/控制单位检查通过。
- 修改红车换道与速度不会同时修改冻结正常蓝车参数。
- 相同训练器在假CARLA world/actor上完成采样、更新和配对验证，确认每次释放场景后才加载下一条件/A/B，未连接真实CARLA。
- 合成多seed负效果表通过汇总检查：负差值和负区间被保留，不强迫新方法胜出。
- 9组短工程集成（正常G0与8组对抗算法）跑通训练/配对评估/汇总及2×2交互输出，记录在runs/research_v3_final；该记录先于最终独立正常蓝车参数/教师资格补丁，不作为后续续训或性能结果。
- 最终代码再运行正常G0、对抗G0/G2/G4各32步、配对selection评估、GIF和汇总：runs/research_v3_releasecheck/receipt.json。只作工程验证，数据规模不足以说明算法优势。

没有启动27组100k主实验，没有读取真实研究的最终测试集，没有操作当前CARLA服务。真实Town05场景载入/控制标定/传感器同步及多seed性能复验尚未执行。

复现检查：

```bash
python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 -v
python -m highwaytest.research validate --config configs/research_v3/attack/G0_s42.json --output runs/new_calibration.json
```

服务器与CARLA实际运行边界分别见RESEARCH_V3.md与ALGORITHM_TRANSFER.md。v3任务、输入和奖励都变化，旧v2结果不能直接改标签或继续权重训练后当v3主实验。
