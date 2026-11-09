# 2026-11-07 工程验证

使用 `D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe`，Highway-env 1.10.2。

|检查|结果|
|---|---|
|`python -m unittest highwaytest.tests.test_contracts -v`|8项通过，包含精确恢复、冻结教师与消息梯度|
|`python -m highwaytest.diagnostics.smoke`|G0–G5 六组全部完成，每组 A+B 合计64步|
|固定模型通信干预|六组均完成 normal/zero/stale/drop 四种评估|
|脚本任务可行性|两类任务各4案例，IDM 蓝车8/8安全到达；不是学习结果|
|CLI train / resume|128/256步在完整边界退出，恢复后完成256/256步|
|CLI evaluate / GIF|成功读取选模 checkpoint，导出 JSON/GIF|
|CLI report|六组普通测试汇总成功，导出 CSV/JSON/PNG|
|compileall / git diff --check|通过|

修正版集成记录：`runs/smoke_20261107_103630/receipt.json`。

- 每组独立保存配置、源码、指标、checkpoint、四种评估报告。
- `comparison/` 保存汇总文件；每组仅一个 seed，标准差为空，不能作稳定性结论。
- `preview.gif` 为G4极短训练后的环境示意。
- CLI 边界恢复记录：`runs/cli_smoke_20261107/summary.json`。
- CLI 评估导出：`runs/cli_evaluation_20261107/`。

短测时间窗刻意缩短，学习模型成功率为0不代表完整任务不可学习；不能把这里的数值当算法性能。正常20s任务只做了脚本可行性探针，尚未完成普通 MAPPO 的多 seed 学习曲线和通信收益验证。下一步从 `configs/driving_g0.json` 开始预实验。
