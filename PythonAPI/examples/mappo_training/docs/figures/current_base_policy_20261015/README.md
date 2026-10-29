# 当前基础策略训练场景地图（2026-10-15）

本图对应当前基础红车训练所用的 **Town05 / R8-A 12 场景池**，不是历史 Town02 的 16 场景，也不是其他 Town05 版本。场景源为 `data/training_scenarios_town05_r8a_v1/`，例如 `configs/base_mappo/p0_base_town05_r8c1_pace_v1.json` 的 `trainer_args.scenario_dir` 指向该池。每次 rollout 的 episode 配额与不同 JSON 的数量不是同一概念。

## 总览与读图说明

![Town05 当前基础训练地图](./town05_current_base_overview.png)

- 左侧为本地 `Town05.xodr` 生成的 Driving 车道路网，突出 Roads 44、3。地图上方是 CARLA 世界坐标的 +Y，右方为 +X；方向名沿用场景标签，不表示地理北。
- 右侧表示各方向任务的所在路段与范围；红点为起点侧，绿点为目标侧，不逐一叠加两辆车的所有出生点。准确的双车布置见下面的逐场景图。
- 细图中 A0 为红色、A1 为橙色，**二者都是基础训练中的红方车辆**，只是为了区分而采用不同颜色；没有添加蓝方对抗车。
- 每张细图统一把行驶方向旋转到向右，采用相同米制比例。绿色矩形为各车实际目标框，尺寸均为 6 × 3.2 m。两车横向中心距约 3.5 m，初始纵向错位不超过 0.01 m。
- 目标采用环境 `apply_scenario_config()` 中 `preserve_start_lane` 的投影规则绘制，不把共享 `goal_zone` 或 JSON 中的显式目标直接当成最终运行时目标。坐标保留两位小数。
- 线和箭头是配置的直行任务示意，**不是实际 rollout 轨迹，也不能证明驾驶能力、训练收敛或策略多样性**。

## 每个 JSON 的配置与细图

速度顺序均为 A0 / A1，距离是出生中心到对应目标中心的近似长度。

| ID | JSON | 阶段 / 方向 | 道路 | 初速 m/s | 距离 / 时限 | 双车细图 |
|---|---|---|---|---|---|---|
| 01 | `scenario_01_north_short_rolling_a0_left.json` | S0-A / 北向 | 44 | 2 / 2 | 15 m / 20 s | [查看](./scene_01.png) |
| 02 | `scenario_02_south_short_rolling_a1_left.json` | S0-A / 南向 | 44 | 2 / 2 | 15 m / 20 s | [查看](./scene_02.png) |
| 03 | `scenario_03_north_short_stationary_a0_left.json` | S0-B / 北向 | 44 | 0 / 0 | 15 m / 20 s | [查看](./scene_03.png) |
| 04 | `scenario_04_south_short_stationary_a1_left.json` | S0-B / 南向 | 44 | 0 / 0 | 15 m / 20 s | [查看](./scene_04.png) |
| 05 | `scenario_05_northbound_free_flow.json` | S1 / 北向 | 44 | 2 / 4 | 45 m / 50 s | [查看](./scene_05.png) |
| 06 | `scenario_06_southbound_free_flow.json` | S1 / 南向 | 44 | 4 / 2 | 45 m / 50 s | [查看](./scene_06.png) |
| 07 | `scenario_07_eastbound_free_flow.json` | S1 / 东向 | 3 | 2 / 4 | 42 m / 50 s | [查看](./scene_07.png) |
| 08 | `scenario_08_westbound_free_flow.json` | S1 / 西向 | 3 | 4 / 2 | 45 m / 50 s | [查看](./scene_08.png) |
| 09 | `scenario_09_north_parallel_stationary_a0_left.json` | S2 / 北向 | 44 | 0 / 0 | 45 m / 50 s | [查看](./scene_09.png) |
| 10 | `scenario_10_south_parallel_stationary_a1_left.json` | S2 / 南向 | 44 | 0 / 0 | 45 m / 50 s | [查看](./scene_10.png) |
| 11 | `scenario_11_east_parallel_stationary_a1_left.json` | S2 / 东向 | 3 | 0 / 0 | 39.5 m / 50 s | [查看](./scene_11.png) |
| 12 | `scenario_12_west_parallel_stationary_a0_left.json` | S2 / 西向 | 3 | 0 / 0 | 39.5 m / 50 s | [查看](./scene_12.png) |

四阶段采用累积纳入：S0-A 共 2 例，S0-B 共 4 例，S1 共 8 例，S2 共 12 例；这表示可用池，不意味着每次 rollout 必然逐例等量采样。

[打开 12 场景高清总图](./town05_current_base_12_scenarios.png)。各单图完整标注 A0/A1 的起点、目标点、yaw、初速、lane ID、任务距离和时限。

## 核验与边界

离线读取本地 CARLA 0.9.16 `Town05.xodr`，校验 12 个 JSON、24 个目标点与各自起点处于相同道路及车道，车道宽度 3.5 m。源 JSON 的生成前后 SHA256 一致，未更改场景、训练参数或运行中的 CARLA 世界；没有新增驾驶试验。

[scenario_manifest.json](./scenario_manifest.json) 保存源文件哈希、道路/车道编号、计算目标和原始场景字段，便于追溯。12 例均近似并排、6 例静止和 6 例滚动，是当前基础直行课程的可视化；不能用旧版本曾包含的跟驰/必需汇入任务来描述这套池。
