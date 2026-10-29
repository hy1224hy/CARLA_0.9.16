# u400 / u600 全面重新评估（2026-10-27）

本次重新实跑192回合：每个固定checkpoint并排32例＋错位验证64例。相同场景、seed、评估源码、配置和阈值已核对；未复用旧逐例结果。
旧报告完整保留，新报告目录：D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training\diagnostics\base_driving_vnext_20261020\long_s42_u600_v1\reevaluation_u400_u600_20261027_v1

|节点|并排到达|验证到达|验证碰撞/越界/道路违规/超时|资格|
|---|---:|---:|---|---|
|u400|32/32|42/64|2/8/12/0|NO_QUALIFIED_CANDIDATE|
|u600|32/32|44/64|2/6/12/0|NO_QUALIFIED_CANDIDATE|

## parallel 逐车指标

|指标|u400 A0|u400 A1|u600 A0|u600 A1|
|---|---:|---:|---:|---:|
|平均速度 km/h|18.240|12.372|24.151|13.392|
|峰值速度 km/h|29.484|27.813|37.013|23.638|
|超15 km/h有效步比例 %|81.866|43.698|82.591|52.609|
|巡航7.2–15 km/h步比例（逐例均值）%|0.000|30.708|0.000|26.184|
|无遮挡低速比例（逐例均值）%|0.000|11.840|0.000|7.168|
|静止起步延迟均值 s|0.675|1.888|0.575|1.663|
|最终路线完成度均值 %|90.861|91.581|91.201|91.350|
|回报均值|30.026|31.180|26.482|30.925|
|刹车步比例均值 %|0.000|8.198|0.000|7.480|
|参考车道偏离逐例P95均值 m|1.259|0.712|1.347|0.654|
|最大参考车道偏离 m|1.746|1.747|1.749|0.951|
|实际转向变化率逐例P95均值 /s|2.035|2.613|2.594|2.840|
|缺终止实际回读案例数|32.000|32.000|32.000|32.000|
|全部硬门槛合格案例数|0.000|0.000|0.000|0.000|

|未过检查的案例数|u400 A0|u400 A1|u600 A0|u600 A1|
|---|---:|---:|---:|---:|
|own_goal_reached|0|0|0|0|
|safe|0|0|0|0|
|stationary_unblocked_startup|0|8|0|8|
|preferred_cruise_speed|32|32|32|32|
|unblocked_low_speed|0|16|0|14|
|upper_task_speed|32|32|32|32|
|reference_lane_stability|30|18|32|20|
|actual_control_coverage|0|0|0|0|
|actual_steering_stability|25|30|32|30|

|节点|分层轴|分层|共同到达|
|---|---|---|---:|
|u400|direction|eastbound|8/8|
|u400|direction|northbound|8/8|
|u400|direction|southbound|8/8|
|u400|direction|westbound|8/8|
|u400|startup_mode|rolling|16/16|
|u400|startup_mode|stationary|16/16|
|u400|role|left_parallel|16/16|
|u400|role|right_parallel|16/16|
|u600|direction|eastbound|8/8|
|u600|direction|northbound|8/8|
|u600|direction|southbound|8/8|
|u600|direction|westbound|8/8|
|u600|startup_mode|rolling|16/16|
|u600|startup_mode|stationary|16/16|
|u600|role|left_parallel|16/16|
|u600|role|right_parallel|16/16|

## validation 逐车指标

|指标|u400 A0|u400 A1|u600 A0|u600 A1|
|---|---:|---:|---:|---:|
|平均速度 km/h|13.694|13.567|18.220|14.341|
|峰值速度 km/h|21.404|28.863|32.873|27.430|
|超15 km/h有效步比例 %|58.114|48.328|70.188|59.500|
|巡航7.2–15 km/h步比例（逐例均值）%|34.346|32.833|17.906|10.077|
|无遮挡低速比例（逐例均值）%|1.190|4.174|0.000|2.614|
|静止起步延迟均值 s|1.184|1.541|0.838|1.253|
|最终路线完成度均值 %|71.204|74.013|74.858|76.538|
|回报均值|3.798|20.963|3.305|22.411|
|刹车步比例均值 %|7.884|3.315|0.533|3.661|
|参考车道偏离逐例P95均值 m|1.301|1.049|1.334|1.054|
|最大参考车道偏离 m|2.120|1.749|2.247|1.749|
|实际转向变化率逐例P95均值 /s|2.104|1.393|3.510|2.157|
|缺终止实际回读案例数|42.000|42.000|44.000|44.000|
|全部硬门槛合格案例数|0.000|0.000|0.000|0.000|

|未过检查的案例数|u400 A0|u400 A1|u600 A0|u600 A1|
|---|---:|---:|---:|---:|
|own_goal_reached|22|22|20|20|
|safe|22|2|20|2|
|stationary_unblocked_startup|8|10|2|8|
|preferred_cruise_speed|47|56|58|64|
|unblocked_low_speed|5|18|0|15|
|upper_task_speed|50|56|58|64|
|reference_lane_stability|58|48|62|40|
|actual_control_coverage|0|0|0|0|
|actual_steering_stability|29|29|42|53|

|节点|分层轴|分层|共同到达|
|---|---|---|---:|
|u400|direction|eastbound|10/16|
|u400|direction|northbound|12/16|
|u400|direction|southbound|10/16|
|u400|direction|westbound|10/16|
|u400|startup_mode|rolling|16/32|
|u400|startup_mode|stationary|26/32|
|u400|role|left_behind|2/16|
|u400|role|right_behind|8/16|
|u400|role|left_ahead|16/16|
|u400|role|right_ahead|16/16|
|u600|direction|eastbound|12/16|
|u600|direction|northbound|10/16|
|u600|direction|southbound|8/16|
|u600|direction|westbound|14/16|
|u600|startup_mode|rolling|18/32|
|u600|startup_mode|stationary|26/32|
|u600|role|left_behind|4/16|
|u600|role|right_behind|8/16|
|u600|role|left_ahead|16/16|
|u600|role|right_ahead|16/16|

## Critic续训前基线

|窗口|联合EV|A0 EV|A1 EV|归一化value loss|原尺度MAE|梯度范数|
|---|---:|---:|---:|---:|---:|---:|
|u541–600|0.282|0.119|0.060|0.101|4.275|0.263|
|u601–660|0.439|0.218|0.149|0.092|4.214|0.299|
|u661–720|0.407|0.227|0.161|0.074|3.480|0.233|
|u721–780|0.359|0.230|0.155|0.084|3.579|0.274|
|u781–800|0.326|0.154|0.156|0.076|3.376|0.206|

解释：冻结驾驶评估衡量Actor表现，不能直接判定Critic收敛。续训将结合固定窗口EV、误差、梯度、归一化和实际驾驶安全共同观察。
缺失终止实际控制回读单独计数，不补零；均值不代替逐例或分层硬门槛。既有走廊的两个seed不代表跨地图泛化。
