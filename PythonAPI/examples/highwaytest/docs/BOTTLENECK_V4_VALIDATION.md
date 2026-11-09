# v4本地验证记录

日期：2026-11-09。Python源码SHA256：c8eb76957dfe6ff393fe31caa4142881190b7d6eda50552200cf359a546ef5ec。

## 自动检查

`python -m unittest highwaytest.tests.test_contracts highwaytest.tests.test_research_v3 highwaytest.tests.test_staged highwaytest.tests.test_bottleneck -v`：64项通过，132.669秒。

覆盖旧协议兼容、真实三车与收窄道路、局部观测/通信信息互补、配对初始状态指纹、通行与停车资格、同一学生输入上的教师蒸馏、续训与TensorBoard记录、源码扫描及CLI校准边界。独立代码审查发现并修复了episode_steps日志字段缺失及旧校准入口误接新协议的问题。

## 完整物理校准

本地目录：`runs/bottleneck_v4_ready_s42`；每场景族8例，共16例，运行四类控制脚本。校准通过；正常IDM C2的合格率100%、违规率0、合格U=0。信息检查确认：改变隐藏红车速度时，蓝1观测变化、蓝2观测不变，而两蓝可通信。

| 控制脚本 | 平均合格U |
|---|---:|
| 正常IDM | 0 |
| 仅物理蓝1低速 | 0.117382 |
| 仅物理蓝2低速 | 0.039100 |
| 两蓝低速 | 0.141463 |

双蓝脚本超过最佳单蓝脚本。这说明当前存在可行的正常通行及联合干扰行为，并具有信息互补结构；脚本未使用学习通信，不能据此声称通信或互学习有效。算法结论需来自统一预算、共同初始化、独立训练种子和独立测试案例的对照实验。

GIF：`runs/bottleneck_v4_ready_s42/calibration.gif`。该GIF是脚本校准，不是训练得到的Mut-MAPPO策略。

## 首轮失败与修正

早期诊断目录`runs/bottleneck_v4_check_s42`的rear_merge间隔34–44m：门控校准通过，但selection正常参考1/8碰撞，BC后位场景碰撞率gate12.5%、selection37.5%，未发布driver.pt。后位正常车辆在汇合时碰撞，参考不可靠，故后位间隔改为60–70m，各组及C0/C1/C2统一修改，无降低安全或通行门槛。

间隔方案固定后，正常IDM在train/gate/selection/test各16个后位案例均合格。这是固定控制器的工程可行性检查；这些test案例不再是完全未查看的案例，但未用学习策略或攻击收益调参。严格完全未触碰的最终主表应另设冻结留出案例。

## 正常驾驶预训练

新源码完整运行64回合恢复驾驶示范、40轮BC。gate与selection的lead_merge/rear_merge四个分层各8例：均合格率100%、违规率0、无效参考0。`bootstrap/driver.pt`已通过资格门槛发布。

随后普通PPO完成1024个A+B决策步，选中A模型eligible=true；导出时重新验证gate/selection，两族合格率均100%，生成同一份`attack_initial.pt`。实际TensorBoard事件文件读取通过：1个文件、552个scalar标签，包含qualified_attack_utility与blue_min_progress_retention。

这里的1024/32步预算仅用于流程检查。服务器主流程仍为100k普通PPO及各组200k对抗PPO，不应把本地短程检查当作收敛实验。

G4从共享初始化完成32个A+B决策步；每族1例的gate/selection以及额外selection配对评价与GIF均正常完成，额外评价合格率100%、违规0、U=0。课程权重仅0.1，selected的eligible=false正确：它只是诊断模型，没有经过全强度对抗训练，不能作为算法优势结果。输出在`diagnostics/attack_G4_smoke`。

发布包只含代码、配置、文档和脚本，不含本地训练权重与runs；解压后验证源码身份及prepare依赖。

本次没有启动真实CARLA；方法接口可供后续适配，不能将Highway结果当作CARLA实测结果。
