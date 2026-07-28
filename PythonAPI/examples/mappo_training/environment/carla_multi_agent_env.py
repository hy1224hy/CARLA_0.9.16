"""
CARLA多智能体环境包装器
支持两个自动驾驶车辆在Town02直道上进行训练
"""

import carla
import numpy as np
import random
import time
from typing import Dict, List, Tuple, Optional
import math
import weakref
from carla import ColorConverter as cc
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

class CarlaMultiAgentEnv:
    """CARLA多智能体环境"""
    
    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 2000,
        town: str = 'Town02',
        num_agents: int = 2,
        straight_road_length: float = 100.0,
        timestep: float = 0.1,
        max_steps: int = 1000,
        spawn_vehicles: bool = True,
        adversarial_agent_ids: Optional[List[int]] = None,
        reward_version: str = 'v2',
        disable_tensorboard: bool = False,
        seed: Optional[int] = None,
        reward_gamma: float = 0.99,
        adversary_collision_ends_episode: bool = True,
    ):
        """
        初始化CARLA多智能体环境

        Args:
            host: CARLA服务器地址
            port: CARLA服务器端口
            town: 地图名称
            num_agents: 智能体数量
            straight_road_length: 直道长度（米）
            timestep: 时间步长（秒）
            max_steps: 最大步数
            adversarial_agent_ids: 对抗智能体ID列表，None表示无对抗智能体
            reward_version: 奖励函数版本 'v1'（原始二元阶梯）或 'v2'（全连续梯度，默认）
            disable_tensorboard: 禁用环境内部TB writer（训练脚本自行管理TB时设为True）
        """
        self.host = host
        self.port = port
        self.town = town
        self.num_agents = num_agents
        self.straight_road_length = straight_road_length
        self.timestep = timestep
        self.max_steps = max_steps
        self.adversarial_agent_ids = adversarial_agent_ids if adversarial_agent_ids is not None else []
        self.reward_version = reward_version
        # 修改1：对抗车(蓝)自撞是否结束整个回合。
        #   True（默认，旧行为）= 任一 agent 碰撞即全局终止；
        #   False（PSRO/eval 用）= 仅合作车(红)碰撞/红车全到达/超时才结束，蓝车自撞只 despawn 该车、
        #     红车继续 —— 消除"蓝车自杀=零成本终止回合剥夺红车成功"的退化漏洞。
        self.adversary_collision_ends_episode = adversary_collision_ends_episode
        self.seed = seed
        # R-1 修复：approach 势能塑形用的折扣因子，应与训练 gamma 一致（默认 0.99）
        self.reward_gamma = reward_gamma

        # ─── P2：观测归一化常量集中化 ───
        # 观测本就归一化到约 [-1,1]（base 8 维 + lidar 11 维），此处把魔法数集中
        # 便于维护，并在 _get_observations 末尾加防御性 clip 兜底极端/越界值。
        self.OBS_POS_SCALE = 100.0    # dx/dy/distance 归一化尺度（米）
        self.OBS_SPEED_SCALE = 30.0   # speed 归一化尺度（m/s）
        self.OBS_CLIP = 5.0           # 防御性裁剪上限（正常范围内不触发，仅兜底）
        
        # 连接CARLA客户端
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        # 加载世界
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()

        # 人行横道多边形缓存（懒加载，主动检测车辆是否压横道）
        self._crosswalk_polys = None
        
        # 设置同步模式
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = timestep
        self.world.apply_settings(settings)
        
        # 获取蓝图库
        self.blueprint_library = self.world.get_blueprint_library()
        
        # 车辆和传感器
        self.vehicles: List[carla.Actor] = []
        self.sensors: List[carla.Actor] = []
        self.camera_sensors: Dict[int, carla.Actor] = {}  # {agent_id: camera_sensor}
        self.camera_images: Dict[int, np.ndarray] = {}  # {agent_id: image_array}
         # 碰撞传感器
        self.collision_sensors = {}  # {agent_id: sensor_actor}
        self.collision_events = {}   # {agent_id: bool} 记录当前步是否发生碰撞
        #车道偏离传感器
        self.lane_sensors = {}  # {agent_id: sensor_actor}
        self.lane_invasion_events = {}   # {agent_id: bool} 记录当前步是否发生车道偏离
        self.crosswalk_invasion_events = {}  # {agent_id: bool} 压人行横道(NONE)，按碰撞处理

        # 状态信息
        self.current_step = 0
        self.straight_waypoints: List[carla.Waypoint] = []
        self.start_waypoints: List[carla.Waypoint] = []
        self.target_waypoints: List[carla.Waypoint] = []
        self.reached_target: Dict[int, bool] = {}
        # 统一 episode 生命周期协议。active_mask 表示下一步仍可接收动作/采样的 agent；
        # terminated/truncated 保留 Gymnasium 语义；episode_end_reason 只描述全局回合。
        self.active_mask: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.last_terminated: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.last_truncated: Dict[int, bool] = {i: False for i in range(self.num_agents)}
        self.episode_end_reason: str = "running"

        # 上一步控制量（用于平滑性惩罚）
        self.prev_controls: Dict[int, Dict[str, float]] = {}

        # 自定义起始位置（优先级高于start_waypoints）
        self.custom_start_transforms: Optional[List[carla.Transform]] = None
        # 自定义终点区域（矩形），None 表示使用直道默认终点
        self.goal_zone: Optional[Dict] = None

        # 找到直道（默认位置，后续可被 JSON 场景配置覆盖）
        self._find_straight_road()

        # 生成车辆（可选）
        if spawn_vehicles:
            self._spawn_vehicles()
        
        # 交通管理器
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

        # 修复 P1：把随机种子贯通到 TrafficManager，保证他车行为可复现。
        # （torch/numpy/random 已由训练脚本 set_seed 播种；TM 是此前唯一未覆盖的非确定源）
        if seed is not None:
            try:
                self.traffic_manager.set_random_device_seed(int(seed))
            except Exception as e:
                print(f"[警告] 无法设置 TrafficManager 随机种子: {e}")


        # TensorBoard记录
        self.use_tensorboard = not disable_tensorboard

        # 全局步数计数器（跨episode不重置，保证TB x轴单调递增）
        self._total_env_steps = 0
        self._episode_count = 0

        if self.use_tensorboard:
            # 获取进程ID和worker ID（用于多进程环境）
            worker_id = os.environ.get('RAY_WORKER_ID', '0')
            process_id = os.getpid()

            # 使用绝对路径，并为每个worker创建独立目录
            base_log_dir = os.path.abspath('logs/carla_training_reward')
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.tb_log_dir = os.path.join(base_log_dir, f"{timestamp}_worker_{worker_id}_pid_{process_id}")

            try:
                os.makedirs(self.tb_log_dir, exist_ok=True)
                self.tb_writer = SummaryWriter(self.tb_log_dir)
                print(f"TensorBoard日志目录: {self.tb_log_dir}")
            except Exception as e:
                print(f"警告: 无法初始化TensorBoard writer: {e}")
                print("继续运行，但不记录TensorBoard日志")
                self.use_tensorboard = False
                self.tb_writer = None
        else:
            self.tb_writer = None

        # 用于记录累计奖励（用于计算平均值）
        self.episode_reward_buffer = {i: [] for i in range(self.num_agents)}
        self.episode_reward_details_buffer = {i: [] for i in range(self.num_agents)}
        # 最新一步的原始奖励诊断。它不参与 reward 计算，只用于训练入口的
        # 版本化指标协议区分“字段缺失 / 设计禁用 / 未触发 / 真实零值”。
        self.last_reward_diagnostics: Dict[int, Dict[str, float]] = {}
        self.episode_step = 0
   

    def _log_to_tensorboard(self, rewards: Dict[int, float], reward_details: Dict[int, Dict], step: int):
        """
        将奖励信息记录到TensorBoard

        Args:
            rewards: 每个agent的总奖励
            reward_details: 每个agent的奖励明细
            step: 当前步数（未使用，保留兼容性；实际用 self._total_env_steps）
        """
        if not self.use_tensorboard or self.tb_writer is None:
            return

        global_step = self._total_env_steps

        for agent_id in range(self.num_agents):
            if agent_id not in rewards or agent_id not in reward_details:
                continue

            total_reward = rewards[agent_id]
            details = reward_details[agent_id]

            # 计算每部分奖励占总奖励的比例（使用绝对值避免负值问题）
            abs_total = sum(abs(v) for v in details.values())

            # 记录总奖励
            self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward/Total', total_reward, global_step)

            # 记录每个奖励分量及其占比
            for key, value in details.items():
                self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward/{key.capitalize()}', value, global_step)
                if abs_total > 0:
                    percentage = (abs(value) / abs_total) * 100.0
                    self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward_Percentage/{key.capitalize()}', percentage, global_step)
                else:
                    self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward_Percentage/{key.capitalize()}', 0.0, global_step)

            reward_dict = {f'{k}': v for k, v in details.items()}
            reward_dict['total'] = total_reward
            self.tb_writer.add_scalars(f'Agent_{agent_id}/Reward_Components', reward_dict, global_step)
    # 新增方法：挂载碰撞传感器
    def _attach_collision_sensor(self, agent_id: int, vehicle: carla.Actor):
        """为车辆附加碰撞传感器"""
        bp = self.blueprint_library.find('sensor.other.collision')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        
        self.collision_sensors[agent_id] = sensor
        self.collision_events[agent_id] = False
        
    # 注册回调：一旦碰撞，将标志位置为 True
        def _on_collision(event):
            self.collision_events[agent_id] = True
    
        sensor.listen(_on_collision)


    # 挂载车道偏离传感器
    def _attach_lane_invasion_sensor(self, agent_id: int, vehicle: carla.Actor):
        """为车辆附加车道偏离传感器"""
        bp = self.blueprint_library.find('sensor.other.lane_invasion')
        sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)
        
        self.lane_sensors[agent_id] = sensor
        self.lane_invasion_events[agent_id] = False
        self.crosswalk_invasion_events[agent_id] = False
        # 添加详细的车道偏离信息存储
        self.lane_invasion_details = getattr(self, 'lane_invasion_details', {})
        self.lane_invasion_details[agent_id] = {
            'has_invasion': False,
            'crossed_markings': [],
            'marking_types': []
        }
        
        # 使用weakref避免循环引用（参考碰撞传感器的实现方式）
        weak_self = weakref.ref(self)
        
        # 回调函数
        def _on_invasion(event):
            """车道偏离事件回调"""
            self = weak_self()
            if not self:
                return
            
            # 记录发生了偏离
            self.lane_invasion_events[agent_id] = True           
            # 获取详细的车道线信息
            if hasattr(event, 'crossed_lane_markings') and event.crossed_lane_markings:
                marking_types = []
                for marking in event.crossed_lane_markings:
                    # 获取车道线类型（实线、虚线等）
                    marking_type = str(marking.type).split('.')[-1] if hasattr(marking, 'type') else 'Unknown'
                    marking_types.append(marking_type)
                
                self.lane_invasion_details[agent_id] = {
                    'has_invasion': True,
                    'crossed_markings': event.crossed_lane_markings,
                    'marking_types': marking_types
                }
                # 若压过 NONE（人行横道），按碰撞处理
                if 'NONE' in marking_types:
                    self.crosswalk_invasion_events[agent_id] = True
                    # print(f"Agent {agent_id} 压人行横道(NONE)，按碰撞处理")
                
                # 打印调试信息（可选，用于验证传感器是否工作）
                # print(f"Agent {agent_id} 车道偏离检测到! 压过的车道线类型: {marking_types}")
            else:
                self.lane_invasion_details[agent_id] = {
                    'has_invasion': True,
                    'crossed_markings': [],
                    'marking_types': []
                }
                # print(f"Agent {agent_id} 车道偏离检测到! (无详细车道线信息)")
        
        sensor.listen(_on_invasion)
        # print(f"Agent {agent_id} 车道偏离传感器已挂载")

    def _find_straight_road(self):
        """在Town02中找到一条直道"""
        # print("正在查找Town02中的直道...")
        
        # 获取所有waypoint
        all_waypoints = self.map.generate_waypoints(2.0)
        
        # 寻找最长的直道段
        best_straight_segment = []
        max_length = 0
        
        # 从多个起点尝试
        spawn_points = self.map.get_spawn_points()
        for start_spawn in spawn_points[:20]:  # 检查前20个spawn点
            try:
                start_waypoint = self.map.get_waypoint(start_spawn.location)
                
                # 沿着道路向前查找直道
                current_waypoint = start_waypoint
                straight_segment = [current_waypoint]
                segment_length = 0.0
                
                for _ in range(int(self.straight_road_length / 2.0)):
                    next_waypoints = current_waypoint.next(2.0)
                    if not next_waypoints:
                        break
                    
                    next_wp = next_waypoints[0]
                    
                    # 检查是否为直道（yaw角度变化小）
                    current_yaw = current_waypoint.transform.rotation.yaw
                    next_yaw = next_wp.transform.rotation.yaw
                    yaw_diff = abs(next_yaw - current_yaw)
                    
                    # 处理角度环绕
                    if yaw_diff > 180:
                        yaw_diff = 360 - yaw_diff
                    
                    # 如果角度变化小于5度，认为是直道
                    if yaw_diff < 5.0:
                        straight_segment.append(next_wp)
                        segment_length += 2.0
                        current_waypoint = next_wp
                    else:
                        break
                
                if segment_length > max_length and segment_length >= self.straight_road_length * 0.8:
                    max_length = segment_length
                    best_straight_segment = straight_segment
                    
            except Exception as e:
                continue
        
        if not best_straight_segment:
            # 如果没找到，使用默认位置（Town02的一个已知直道位置）
            # print("使用默认直道位置...")
            default_location = carla.Location(x=100, y=-257, z=0.5)
            start_waypoint = self.map.get_waypoint(default_location)
            best_straight_segment = [start_waypoint]
            
            current_waypoint = start_waypoint
            for _ in range(int(self.straight_road_length / 2.0)):
                next_waypoints = current_waypoint.next(2.0)
                if next_waypoints:
                    best_straight_segment.append(next_waypoints[0])
                    current_waypoint = next_waypoints[0]
                else:
                    break
        
        self.straight_waypoints = best_straight_segment
        # print(f"找到直道，长度: {len(self.straight_waypoints) * 2.0} 米")
        
        # 设置起始和目标waypoint
        if len(self.straight_waypoints) >= self.num_agents:
            total_wp = len(self.straight_waypoints)
            # 按角色分配位置：合作车在道路前段（20%~50%），对抗车穿插其间
            cooperative_ids = [i for i in range(self.num_agents) if i not in self.adversarial_agent_ids]
            adversarial_ids = self.adversarial_agent_ids

            # 为每个agent计算waypoint索引
            # 确保任意两车之间至少有 min_gap 个waypoint（每个2m，即至少保持间距）
            min_gap = 5  # 至少10米间距
            agent_indices = {}
            if cooperative_ids and adversarial_ids:
                total_agents_count = len(cooperative_ids) + len(adversarial_ids)
                # 所有车辆均匀分布在 20%~60% 位置，按agent_id交错排列
                # 合作车在前，对抗车在后，各自留有足够间距
                for ci, aid in enumerate(sorted(cooperative_ids)):
                    agent_indices[aid] = int(total_wp * (0.20 + ci * min_gap / total_wp))
                # 对抗车从合作车后方开始放置
                adv_start = int(total_wp * 0.40)
                for ai, aid in enumerate(sorted(adversarial_ids)):
                    agent_indices[aid] = int(total_wp * (0.40 + ai * min_gap / total_wp))
                # 检查是否有碰撞索引，重新均匀分配
                all_indices = list(agent_indices.values())
                if len(set(all_indices)) < len(all_indices):
                    # 有重复索引，均匀分配全部agent
                    step = max(1, total_wp // (total_agents_count + 1))
                    all_ids = sorted(list(range(self.num_agents)))
                    for i, aid in enumerate(all_ids):
                        agent_indices[aid] = min((i + 1) * step, total_wp - 1)
            else:
                # 无角色区分时均匀分布
                step = max(1, total_wp // (self.num_agents + 1))
                for i in range(self.num_agents):
                    agent_indices[i] = min((i + 1) * step, total_wp - 1)

            for i in range(self.num_agents):
                idx = min(agent_indices[i], total_wp - 1)
                self.start_waypoints.append(self.straight_waypoints[idx])

            print(f"智能体位置分配: { {i: f'{agent_indices[i]}/{total_wp}' for i in range(self.num_agents)} }")

            # 目标位置是道路的末端
            self.target_waypoints = [self.straight_waypoints[-1]] * self.num_agents
        else:
            # 如果waypoint不够，使用同一个起点
            for _ in range(self.num_agents):
                self.start_waypoints.append(self.straight_waypoints[0])
                self.target_waypoints.append(self.straight_waypoints[-1])
    
    # ── 目标区域辅助方法 ──

    def set_goal_zone(self, goal_config: Dict):
        """
        设置终点矩形区域（覆盖 _find_straight_road 的默认终点）

        goal_config 格式: {
            "x": 0.0, "y": 300.0, "z": 0.5,
            "yaw": 90.0,
            "half_length": 3.0,
            "half_width": 1.75
        }
        """
        import carla
        center = carla.Location(
            x=float(goal_config["x"]),
            y=float(goal_config["y"]),
            z=float(goal_config.get("z", 0.5)),
        )
        yaw = float(goal_config["yaw"])
        yaw_rad = math.radians(yaw)
        forward = carla.Vector3D(math.cos(yaw_rad), math.sin(yaw_rad), 0.0)
        # 右向量 = 前进方向逆时针旋转90°
        right = carla.Vector3D(-math.sin(yaw_rad), math.cos(yaw_rad), 0.0)

        self.goal_zone = {
            "center": center,
            "yaw": yaw,
            "forward": forward,
            "right": right,
            "half_length": float(goal_config.get("half_length", 3.0)),
            "half_width": float(goal_config.get("half_width", 1.75)),
        }
        print(f"终点区域已设置: center=({center.x:.1f},{center.y:.1f}), "
              f"yaw={yaw:.1f}°, size={self.goal_zone['half_length']*2:.1f}m×{self.goal_zone['half_width']*2:.1f}m")

    def _get_target_location(self, agent_id: int) -> carla.Location:
        """返回 agent_id 的目标位置（优先 goal_zone，回退 target_waypoints）"""
        if self.goal_zone is not None:
            return self.goal_zone["center"]
        if agent_id < len(self.target_waypoints) and self.target_waypoints[agent_id] is not None:
            return self.target_waypoints[agent_id].transform.location
        return carla.Location(x=0, y=0, z=0)

    def _ensure_crosswalk_polygons(self):
        """懒加载并缓存当前 map 的人行横道多边形。

        carla 的 map.get_crosswalks() 返回扁平的 Location 列表：每个横道是
        若干顶点构成的闭合多边形，闭合处会重复第一个顶点。
        """
        if self._crosswalk_polys is not None:
            return
        self._crosswalk_polys = []
        try:
            raw = self.map.get_crosswalks() or []
        except Exception:
            return
        current: List[Tuple[float, float]] = []
        for pt in raw:
            xy = (float(pt.x), float(pt.y))
            if current and abs(xy[0] - current[0][0]) < 0.05 and abs(xy[1] - current[0][1]) < 0.05:
                if len(current) >= 3:
                    self._crosswalk_polys.append(current)
                current = []
            else:
                current.append(xy)
        if len(current) >= 3:
            self._crosswalk_polys.append(current)

    @staticmethod
    def _point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
        """射线法判断点是否在多边形内。"""
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            denom = (yj - yi) if abs(yj - yi) > 1e-12 else 1e-12
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / denom + xi):
                inside = not inside
            j = i
        return inside

    def _vehicle_on_crosswalk(self, vehicle: carla.Actor) -> bool:
        """车辆中心是否压在任意人行横道多边形内（主动检测，补偿 lane_invasion 传感器漏报）。"""
        self._ensure_crosswalk_polygons()
        if not self._crosswalk_polys:
            return False
        try:
            loc = vehicle.get_transform().location
        except Exception:
            return False
        x, y = float(loc.x), float(loc.y)
        for poly in self._crosswalk_polys:
            # 用 bbox 先做粗筛，加速
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if x < min(xs) or x > max(xs) or y < min(ys) or y > max(ys):
                continue
            if self._point_in_polygon(x, y, poly):
                return True
        return False

    def _is_in_goal_zone(self, vehicle_location: carla.Location) -> bool:
        """检查车辆位置是否在终点矩形区域内"""
        if self.goal_zone is not None:
            gz = self.goal_zone
            rel = carla.Vector3D(
                vehicle_location.x - gz["center"].x,
                vehicle_location.y - gz["center"].y,
                0.0,
            )
            x_along = rel.x * gz["forward"].x + rel.y * gz["forward"].y
            y_lat = rel.x * gz["right"].x + rel.y * gz["right"].y
            return abs(x_along) <= gz["half_length"] and abs(y_lat) <= gz["half_width"]

        # 回退：使用 target_waypoints + 默认尺寸
        return False  # 由调用方处理（保持兼容 _check_dones 的默认逻辑）

    def apply_scenario_config(self, config: Dict):
        """
        应用完整场景配置（起点 + 终点）

        config 格式: {
            "town": "Town02",          # 可选
            "goal_zone": { ... },      # 可选
            "agents": {                # 可选
                "agent0": {"x": ..., "y": ..., "z": 0.5, "yaw": 90.0},
                ...
            }
        }
        兼容旧格式（顶层直接 agent0 而无 agents 嵌套）。
        """
        # 1. 终点区域
        if "goal_zone" in config:
            self.set_goal_zone(config["goal_zone"])

        # 2. 起始位置：兼容 "agents" 嵌套和旧格式
        agents_dict = config.get("agents", config)
        # 过滤掉非 agent 的顶层 key（description, town, goal_zone 等）
        agent_keys = {k for k in agents_dict if k.startswith("agent")}
        if agent_keys:
            self.set_custom_start_positions({k: agents_dict[k] for k in agent_keys})

        # print("场景配置已应用")

    # ── 车辆生成 ──

    def _spawn_vehicles(self):
        """生成车辆"""
        # 清理现有车辆
        self._cleanup()

        for i in range(self.num_agents):
            is_adversarial = i in self.adversarial_agent_ids
            vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
            # 自车红色，对抗车蓝色
            vehicle_bp.set_attribute('color', '0,0,255' if is_adversarial else '255,0,0')

            # 优先使用自定义位置，否则使用start_waypoints
            if self.custom_start_transforms and i < len(self.custom_start_transforms) and self.custom_start_transforms[i] is not None:
                spawn_transform = self.custom_start_transforms[i]
                # print(f"车辆 {i} 使用自定义位置")
            elif i < len(self.start_waypoints):
                spawn_transform = self.start_waypoints[i].transform
            else:
                # raise RuntimeError(f"无法获取车辆 {i} 的起始位置")
                pass

            # 尝试生成车辆
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)

            # 碰撞时尝试多种偏移
            retry_offsets = [
                (0, 0, 0.5),       # 垂直
                (1.5, 0, 0),       # 横向右
                (-1.5, 0, 0),      # 横向左
                (0, 1.0, 0.3),     # 前移+抬高
                (0, -1.0, 0.3),    # 后移+抬高
            ]
            retry_idx = 0
            while vehicle is None and retry_idx < len(retry_offsets):
                dx, dy, dz = retry_offsets[retry_idx]
                retry_transform = carla.Transform(
                    carla.Location(
                        x=spawn_transform.location.x + dx,
                        y=spawn_transform.location.y + dy,
                        z=spawn_transform.location.z + dz,
                    ),
                    spawn_transform.rotation
                )
                vehicle = self.world.try_spawn_actor(vehicle_bp, retry_transform)
                retry_idx += 1

            if vehicle:
                self.vehicles.append(vehicle)
                print(f"车辆 {i} 生成成功，位置: {spawn_transform.location}")
                # 挂载车道偏离传感器
                self._attach_lane_invasion_sensor(i, vehicle)
                # 挂载摄像头
                self._attach_camera(i, vehicle)
                # 挂载碰撞传感器
                self._attach_collision_sensor(i, vehicle)
            else:
                raise RuntimeError(f"无法生成车辆 {i}")

        # 等待世界更新
        self.world.tick()
        time.sleep(0.5)
    
    def _attach_camera(self, agent_id: int, vehicle: carla.Actor):
        """
        为车辆附加摄像头传感器
        
        Args:
            agent_id: 智能体ID
            vehicle: 车辆actor
        """
        # 摄像头蓝图
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '640')
        camera_bp.set_attribute('image_size_y', '480')
        camera_bp.set_attribute('fov', '110')
        
        # 摄像头位置（车辆前方）
        bound_x = 0.5 + vehicle.bounding_box.extent.x
        bound_y = 0.5 + vehicle.bounding_box.extent.y
        bound_z = 0.5 + vehicle.bounding_box.extent.z
        camera_transform = carla.Transform(
            carla.Location(x=+0.8*bound_x, y=+0.0*bound_y, z=1.3*bound_z),
            carla.Rotation(pitch=0.0)
        )
        
        # 生成摄像头
        camera = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=vehicle,
            attachment_type=carla.AttachmentType.Rigid
        )
        
        # 存储摄像头引用
        self.camera_sensors[agent_id] = camera
        self.camera_images[agent_id] = None
        
        # 设置回调函数
        weak_self = weakref.ref(self)
        camera.listen(lambda image: self._parse_image(weak_self, agent_id, image))
        
        # print(f"车辆 {agent_id} 摄像头传感器已附加")
    
    @staticmethod
    def _parse_image(weak_self, agent_id: int, image):
        """解析摄像头图像"""
        self = weak_self()
        if not self:
            return
        
        try:
            # 转换图像
            image.convert(cc.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # 只取RGB，忽略Alpha
            # 注意：CARLA图像已经是RGB格式，不需要BGR转RGB
            
            # 存储图像
            if agent_id in self.camera_images:
                self.camera_images[agent_id] = array
        except Exception as e:
            # print(f"警告: 解析摄像头图像失败 (Agent {agent_id}): {e}")
            pass
    
    def _cleanup(self):
        """清理所有车辆和传感器"""
        # 清理摄像头
        for agent_id, camera in self.camera_sensors.items():
            if camera.is_alive:
                camera.destroy()
        self.camera_sensors.clear()
        self.camera_images.clear()

        # 清理碰撞传感器
        for agent_id, sensor in self.collision_sensors.items():
            if sensor.is_alive:
                sensor.destroy()
        self.collision_sensors.clear()
        self.collision_events.clear()

        # 清理车道偏离传感器
        for agent_id, sensor in self.lane_sensors.items():
            if sensor.is_alive:
                sensor.destroy()
        self.lane_sensors.clear()
        self.lane_invasion_events.clear()
        self.crosswalk_invasion_events.clear()
        
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.destroy()
        self.sensors.clear()
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.vehicles.clear()
        
        self.world.tick()
    
    def get_camera_images(self) -> Dict[int, np.ndarray]:
        """
        获取所有车辆的摄像头图像

        Returns:
            camera_images: {agent_id: image_array}
        """
        return self.camera_images.copy()

    def set_custom_start_positions(self, positions_dict: Dict):
        """
        设置自定义车辆起始位置（下次reset时生效）

        Args:
            positions_dict: 格式为 {"agent0": {"x": 100, "y": -257, "z": 0.5, "yaw": 0.0}, ...}
                           如果不指定 yaw，会自动对齐道路方向
        """
        import carla

        # print(f"正在设置自定义起始位置...")
        self.custom_start_transforms = []

        for i in range(self.num_agents):
            agent_key = f"agent{i}"
            if agent_key in positions_dict:
                pos = positions_dict[agent_key]
                # 创建自定义transform
                location = carla.Location(
                    x=float(pos.get("x", 0.0)),
                    y=float(pos.get("y", 0.0)),
                    z=float(pos.get("z", 0.5))
                )

                # 检查是否显式指定了yaw
                if "yaw" in pos:
                    rotation = carla.Rotation(
                        pitch=float(pos.get("pitch", 0.0)),
                        yaw=float(pos.get("yaw", 0.0)),
                        roll=float(pos.get("roll", 0.0))
                    )
                    yaw_source = "用户指定"
                else:
                    # 没有指定yaw，自动获取道路方向
                    try:
                        waypoint = self.map.get_waypoint(
                            location,
                            project_to_road=True,
                            lane_type=carla.LaneType.Driving
                        )
                        if waypoint:
                            rotation = waypoint.transform.rotation
                            yaw_source = "自动对齐道路"
                        else:
                            rotation = carla.Rotation()
                            yaw_source = "默认(0)"
                    except:
                        rotation = carla.Rotation()
                        yaw_source = "默认(0)"

                transform = carla.Transform(location, rotation)
                self.custom_start_transforms.append(transform)
                print(f"  智能体 {i} 起始位置已设置: x={location.x:.2f}, y={location.y:.2f}, z={location.z:.2f}, yaw={rotation.yaw:.2f} ({yaw_source})")
            elif i < len(self.start_waypoints):
                # 没有自定义位置的智能体，使用默认位置
                self.custom_start_transforms.append(self.start_waypoints[i].transform)
            else:
                self.custom_start_transforms.append(None)


    def _get_ray_detection(self, vehicle: carla.Actor, ray_length: float = 4.0, num_rays: int = 11) -> np.ndarray:
        """
        模拟激光雷达探测（射线检测）

        Args:
            vehicle: 车辆对象
            ray_length: 射线最大探测距离（米）
            num_rays: 射线数量（覆盖全周360度）

        Returns:
            distances: 归一化的距离数组 (shape: [num_rays])
                       1.0 表示范围内无障碍，0.0 表示紧贴障碍
        """
        transform = vehicle.get_transform()
        start_loc = transform.location
        start_loc.z += 1.0  # 抬高起点，避免射到地面

        # 车辆朝向（弧度）
        yaw = math.radians(transform.rotation.yaw)

        # 定义射线角度范围：全周360度
        fov = 2 * math.pi  # 360度视野
        angles = np.linspace(-fov/2, fov/2, num_rays)

        # ─── 自过滤 ───
        # 原版 `detection.label == vehicle.id` 比较的是 CityObjectLabel 枚举（小整数）
        # 与 actor id，永远不匹配 → 每条 ray 都打到自己的车身 → 全 0 距离
        # 修复：用 bounding box 推算"车体最大外接圆半径"，detection 在此距离内直接丢弃
        try:
            extent = vehicle.bounding_box.extent
            self_radius = math.sqrt(extent.x ** 2 + extent.y ** 2) + 0.15
        except Exception:
            self_radius = 1.5  # 兜底：典型轿车半对角线 ≈ 1.35m

        distances = []

        for angle in angles:
            ray_yaw = yaw + angle
            end_loc = carla.Location(
                x=start_loc.x + ray_length * math.cos(ray_yaw),
                y=start_loc.y + ray_length * math.sin(ray_yaw),
                z=start_loc.z
            )

            detections = self.world.cast_ray(start_loc, end_loc)

            det_dist = ray_length

            if detections:
                for detection in detections:
                    dist = start_loc.distance(detection.location)
                    # 过滤自身：在车体半径内的命中点跳过
                    if dist < self_radius:
                        continue
                    if dist < det_dist:
                        det_dist = dist

            # 归一化：1.0=无障碍，0.0=紧贴障碍
            norm_dist = det_dist / ray_length
            distances.append(norm_dist)

        return np.array(distances, dtype=np.float32)
    
    def reset(self) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        重置环境
        
        Returns:
            observations: 每个智能体的观察
            info: 额外信息
        """

        # 如果上一轮有数据，记录episode平均奖励到TensorBoard
        if self.use_tensorboard and self.tb_writer is not None:
            if any(self.episode_reward_buffer.values()):
                for agent_id in range(self.num_agents):
                    if self.episode_reward_buffer[agent_id]:
                        avg_reward = np.mean(self.episode_reward_buffer[agent_id])
                        self.tb_writer.add_scalar(f'Agent_{agent_id}/Episode_Reward/Average',
                                                 avg_reward, self._total_env_steps)
                        
                        # 记录episode平均奖励明细
                        if self.episode_reward_details_buffer[agent_id]:
                            avg_details = {}
                            for key in ["progress", "speed", "centering", "yaw",
                                       "jerk", "steer_rate", "safety_lidar", "ttc",
                                       "collision", "lane_invasion", "step",
                                       "success", "time_bonus"]:
                                values = [d[key] for d in self.episode_reward_details_buffer[agent_id] 
                                         if key in d]
                                if values:
                                    avg_details[key] = np.mean(values)
                            
                            for key, value in avg_details.items():
                                self.tb_writer.add_scalar(f'Agent_{agent_id}/Episode_Reward/{key.capitalize()}',
                                                         value, self._total_env_steps)
                
                # 清空缓冲区
                for agent_id in range(self.num_agents):
                    self.episode_reward_buffer[agent_id] = []
                    self.episode_reward_details_buffer[agent_id] = []
        
        self.episode_step += 1
        self._episode_count += 1
        self.current_step = 0
        
        # 重新生成车辆
        self._spawn_vehicles()

        # 初始化上一帧的控制量（用于平滑性惩罚，首帧无惩罚）
        self.prev_controls = {}

        # 初始化上一帧的距离
        self.prev_distances = {}
        # 初始化回合起始距离（用于"接近终点"连续正向势能奖励，见 _compute_rewards_v2）
        self.initial_distances = {}

        # 初始化是否到达目标
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                target_loc = self._get_target_location(i)
                curr_loc = vehicle.get_location()
                d0 = curr_loc.distance(target_loc)
                self.prev_distances[i] = d0
                self.initial_distances[i] = max(d0, 1.0)  # 防止除零
                self.reached_target[i] = False
            else:
                self.prev_distances[i] = 0.0
                self.initial_distances[i] = 1.0
                self.reached_target[i] = False

        # reset 也返回完整协议，训练侧从第一步起只依赖 active_mask，不再从零观察猜测存活状态。
        self.active_mask = {
            i: bool(i < len(self.vehicles) and self.vehicles[i] is not None
                    and self.vehicles[i].is_alive)
            for i in range(self.num_agents)
        }
        self.last_terminated = {i: False for i in range(self.num_agents)}
        self.last_truncated = {i: False for i in range(self.num_agents)}
        self.episode_end_reason = "running"
        
        # 获取初始观察
        observations = self._get_observations()
        info = {
            'step': self.current_step,
            'active_mask': dict(self.active_mask),
            'terminated': dict(self.last_terminated),
            'truncated': dict(self.last_truncated),
            'episode_end_reason': self.episode_end_reason,
            'pre_collision_kinematics': {},
        }
        
        return observations, info
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """获取所有智能体的观察"""
        observations = {}
        # 设定雷达参数
        NUM_RAYS = 11      # 射线数量
        RAY_LENGTH = 4.0   # 探测距离（全周360度）
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                # 8个基础状态 + NUM_RAYS个雷达状态
                observations[i] = np.zeros(8 + NUM_RAYS, dtype=np.float32)
                continue
            
            # 获取车辆状态
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            control = vehicle.get_control()
            
            # 计算到目标的方向
            target_location = self._get_target_location(i)
            
            # 车辆位置
            vehicle_location = transform.location
            
            # 计算相对位置和方向
            dx = target_location.x - vehicle_location.x
            dy = target_location.y - vehicle_location.y
            
            # 计算距离
            distance = math.sqrt(dx**2 + dy**2)
            
            # 计算角度差（相对于车辆朝向）
            vehicle_yaw = math.radians(transform.rotation.yaw)
            target_yaw = math.atan2(dy, dx)
            angle_diff = target_yaw - vehicle_yaw
            
            # 归一化角度到[-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi
            
            # 速度
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # 当前控制输入
            throttle = control.throttle
            steer = control.steer
            brake = control.brake
            
            # 观察向量: [dx, dy, distance, angle_diff, speed, throttle, steer, brake]
            # P2：使用集中化的归一化常量（throttle/steer/brake 本就在 [0,1]/[-1,1]）
            base_obs = np.array([
                dx / self.OBS_POS_SCALE,
                dy / self.OBS_POS_SCALE,
                distance / self.OBS_POS_SCALE,
                angle_diff / math.pi,          # 归一化到 [-1, 1]
                speed / self.OBS_SPEED_SCALE,  # 归一化速度
                throttle,
                steer,
                brake
            ], dtype=np.float32)

            # 获取雷达感知信息（已归一化到 [0,1]）
            lidar_obs = self._get_ray_detection(vehicle, ray_length=RAY_LENGTH, num_rays=NUM_RAYS)

            # 拼接观测向量 (8 + 11 = 19维)
            obs = np.concatenate([base_obs, lidar_obs])

            # P2：防御性裁剪 —— 正常范围（约 [-1,1]）内不触发，仅兜底长路/越界/数值异常，
            # 避免极端输入直接进网络导致 Tanh 首层饱和或 NaN 传播。
            obs = np.clip(obs, -self.OBS_CLIP, self.OBS_CLIP).astype(np.float32)

            observations[i] = obs

        return observations

    def _snapshot_kinematics(self) -> Dict[int, Dict]:
        """读取当前存活车辆的纯数据运动学快照，不向 info 泄露 CARLA actor。"""
        snapshots: Dict[int, Dict] = {}
        for agent_id, vehicle in enumerate(self.vehicles):
            if vehicle is None or not vehicle.is_alive:
                continue
            try:
                transform = vehicle.get_transform()
                velocity = vehicle.get_velocity()
                angular_velocity = vehicle.get_angular_velocity()
                control = vehicle.get_control()
            except (RuntimeError, AttributeError):
                # actor 可能在异步回调与本线程之间刚好失效；该 agent 不写入不完整快照。
                continue

            speed = math.sqrt(
                velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2
            )
            snapshots[agent_id] = {
                'location': {
                    'x': float(transform.location.x),
                    'y': float(transform.location.y),
                    'z': float(transform.location.z),
                },
                'rotation': {
                    'pitch': float(transform.rotation.pitch),
                    'yaw': float(transform.rotation.yaw),
                    'roll': float(transform.rotation.roll),
                },
                'velocity': {
                    'x': float(velocity.x),
                    'y': float(velocity.y),
                    'z': float(velocity.z),
                },
                'angular_velocity': {
                    'x': float(angular_velocity.x),
                    'y': float(angular_velocity.y),
                    'z': float(angular_velocity.z),
                },
                'control': {
                    'throttle': float(control.throttle),
                    'steer': float(control.steer),
                    'brake': float(control.brake),
                },
                'speed': float(speed),
                'alive': True,
                'reached': bool(self.reached_target.get(agent_id, False)),
            }
        return snapshots
    
    def step(self, actions: Dict[int, np.ndarray]) -> Tuple[Dict[int, np.ndarray], Dict[int, float], Dict[int, bool], Dict]:
        """
        执行一步动作
        
        Args:
            actions: 每个智能体的动作 [throttle, steer, brake]
                    throttle和brake在[0, 1]，steer在[-1, 1]
        
        Returns:
            observations: 新的观察
            rewards: 每个智能体的奖励
            dones: 每个智能体是否完成
            info: 额外信息
        """
        # 应用动作
        # 应用动作 - 如果车辆已经到达终点，不应用动作（让车辆停止）
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive and i in actions and not self.reached_target.get(i, False):
                action = actions[i]
                # 确保动作在有效范围内
                throttle = np.clip(action[0], 0.0, 1.0)
                steer = np.clip(action[1], -1.0, 1.0)
                brake = np.clip(action[2], 0.0, 1.0)
                
                control = carla.VehicleControl(
                    throttle=float(throttle),
                    steer=float(steer),
                    brake=float(brake)
                )
                vehicle.apply_control(control)
            elif vehicle.is_alive and self.reached_target.get(i, False):
                # 到点后强制停止 + 清零物理速度/角速度，避免惯性右滑或缓慢转头
                # 单靠 brake=1 需要好几个 tick 才能让速度降到 0，期间转向轮残量
                # 会让车体侧滑或缓慢转向；直接覆盖刚体的线/角速度才能立刻停稳。
                control = carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0,
                    hand_brake=True,
                )
                vehicle.apply_control(control)
                try:
                    vehicle.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    vehicle.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass
        
        # 保存推进物理世界之前的同帧状态。若本 tick 发生碰撞，这份快照用于终局归因；
        # 尤其是蓝车会在 step 返回前 despawn，训练侧不能再读取其 CARLA actor。
        pre_tick_kinematics = self._snapshot_kinematics()

        # 更新世界
        self.world.tick()
        self.current_step += 1
        self._total_env_steps += 1

        # 主动检测人行横道：lane_invasion 传感器只在跨线那一刻回调，
        # 且 marking_type 是否为 'NONE' 取决于地图标注，常有漏报。
        # 用 map.get_crosswalks() 多边形重叠判定补充，确保已经上人行道立刻触发终止
        for i, _v in enumerate(self.vehicles):
            if _v.is_alive and not self.reached_target.get(i, False):
                if self._vehicle_on_crosswalk(_v):
                    self.crosswalk_invasion_events[i] = True

        # 必须在 _check_dones() 可能 despawn 到点车辆之前固化碰撞标志。
        step_collisions = self._check_collisions()
        pre_collision_kinematics = (
            pre_tick_kinematics if any(step_collisions.values()) else {}
        )

        # 获取新观察
        observations = self._get_observations()
        
        # 计算奖励
        rewards, reward_details = self._compute_rewards()
        reward_diagnostics = {
            agent_id: dict(values)
            for agent_id, values in getattr(
                self, "last_reward_diagnostics", {}
            ).items()
        }

        # 累积到episode缓冲区（用于计算episode平均值）
        if self.use_tensorboard:
            for agent_id in range(self.num_agents):
                if agent_id in rewards:
                    self.episode_reward_buffer[agent_id].append(rewards[agent_id])
                if agent_id in reward_details:
                    self.episode_reward_details_buffer[agent_id].append(reward_details[agent_id])

        

        # 记录到TensorBoard（x轴使用全局步数，确保单调递增）
        self._log_to_tensorboard(rewards, reward_details, self._total_env_steps)
        
        # 检查是否完成（这会更新reached_target状态）
        dones = self._check_dones()
        
        info = {
            'step': self.current_step,
            'reward_details': reward_details,
            'reward_diagnostics': reward_diagnostics,
            'collisions': step_collisions,
            # 修复 #2：区分真正终止与时间截断，供训练侧 GAE 决定是否 bootstrap
            'terminated': dict(getattr(self, 'last_terminated', {})),
            'truncated': dict(getattr(self, 'last_truncated', {})),
            'active_mask': dict(getattr(self, 'active_mask', {})),
            'episode_end_reason': getattr(self, 'episode_end_reason', 'running'),
            # 碰撞 tick 返回推进前的全体存活车辆快照，保证几何归因使用同一时刻的数据。
            'pre_collision_kinematics': pre_collision_kinematics,
        }

        # 修改1：对抗车(蓝)自撞且回合不全局终止时，移除该蓝车（避免撞毁残骸停在单车道里
        # 变成"自杀路障"，那等于换个形式恢复漏洞）。此处在 info 已记录本步 collisions/terminated
        # 之后执行，故蓝车的撞毁仍被训练侧正确捕获（reward 含碰撞罚、terminated=True）。
        if not self.adversary_collision_ends_episode:
            for i in self.adversarial_agent_ids:
                if (self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False)) \
                        and not self.reached_target.get(i, False):
                    if i < len(self.vehicles) and self.vehicles[i] is not None and self.vehicles[i].is_alive:
                        self._despawn_reached_vehicle(i)  # 通用 despawn（不标记 reached_target）

        for i in self.collision_events: # 重置碰撞状态，避免重复惩罚
            self.collision_events[i] = False

        for i in self.lane_invasion_events: # 重置车道偏离状态，避免重复惩罚
            self.lane_invasion_events[i] = False
        for i in self.crosswalk_invasion_events:
            self.crosswalk_invasion_events[i] = False

        # 保存当前控制量，供下一帧平滑性惩罚使用
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                ctrl = vehicle.get_control()
                self.prev_controls[i] = {
                    'throttle': ctrl.throttle,
                    'steer': ctrl.steer,
                    'brake': ctrl.brake,
                }

        return observations, rewards, dones, info


    def _compute_rewards(self) -> Tuple[Dict[int, float], Dict[int, Dict]]:
        """奖励函数调度器"""
        if self.reward_version == 'v1':
            return self._compute_rewards_v1()
        else:
            return self._compute_rewards_v2()

    def _compute_rewards_v1(self) -> Tuple[Dict[int, float], Dict[int, Dict]]:
        """原始二元阶梯奖励（v1，用于 A/B 对比）"""
        rewards = {}
        reward_details = {}
        reward_diagnostics = {}

        def _empty_diagnostics(**overrides):
            values = {
                "progress_m": 0.0,
                "speed_mps": 0.0,
                "lateral_distance_m": 0.0,
                "yaw_error_rad": 0.0,
                "collision_event": 0.0,
                "lane_invasion_sensor_event": 0.0,
                "success_event": 0.0,
            }
            values.update({key: float(value) for key, value in overrides.items()})
            return values

        for i, vehicle in enumerate(self.vehicles):
            # 已到达终点（含 disappear-at-target 被移除的车）：给 0，必须在 not-alive 之前判定
            if self.reached_target.get(i, False):
                rewards[i] = 0.0
                reward_details[i] = {
                    "progress": 0.0, "speed": 0.0, "centering": 0.0, "yaw": 0.0,
                    "collision": 0.0, "lane_invasion": 0.0, "step": 0.0, "success": 0.0,
                }
                reward_diagnostics[i] = _empty_diagnostics()
                continue
            if not vehicle.is_alive:
                rewards[i] = -100.0
                reward_details[i] = {
                    "progress": 0.0, "speed": 0.0, "centering": 0.0, "yaw": 0.0,
                    "collision": -100.0, "lane_invasion": 0.0, "step": 0.0, "success": 0.0,
                }
                reward_diagnostics[i] = _empty_diagnostics(collision_event=1.0)
                continue

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            target_loc = self._get_target_location(i)

            # 1. 进度奖励
            current_dist = transform.location.distance(target_loc)
            prev_dist = self.prev_distances.get(i, current_dist)
            progress = prev_dist - current_dist
            progress_reward = progress * 20.0
            self.prev_distances[i] = current_dist

            # 2. 三段式速度奖励
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            speed_kmh = speed * 3.6
            if speed_kmh < 1.0:
                speed_reward = -0.1
            elif speed_kmh > 25.0:
                speed_reward = -1.0
            else:
                speed_reward = 0.1

            # 3. 二元居中奖励
            current_waypoint = self.map.get_waypoint(transform.location)
            centering_reward = 0.0
            lateral_dist = 0.0
            if current_waypoint:
                veh_loc = transform.location
                wp_loc = current_waypoint.transform.location
                vec_wp_to_veh = carla.Vector3D(veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                right_vector = current_waypoint.transform.get_right_vector()
                lateral_dist = abs(vec_wp_to_veh.x * right_vector.x + vec_wp_to_veh.y * right_vector.y)
                if lateral_dist < 0.05:
                    centering_reward = 0.5

            # 4. 二元航向奖励
            dx = target_loc.x - transform.location.x
            dy = target_loc.y - transform.location.y
            target_yaw = math.atan2(dy, dx)
            vehicle_yaw = math.radians(transform.rotation.yaw)
            angle_diff = abs(target_yaw - vehicle_yaw)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            yaw_reward = 0.1 if angle_diff < 0.2 else 0.0

            # 5. 碰撞惩罚
            crosswalk = self.crosswalk_invasion_events.get(i, False)
            collision_penalty = -100.0 if (self.collision_events.get(i, False) or crosswalk) else 0.0

            # 6. 统一车道偏离惩罚
            sensor_invasion = self.lane_invasion_events.get(i, False)
            if crosswalk:
                lane_invasion_penalty = 0.0
            elif sensor_invasion:
                lane_invasion_penalty = -5.0
            else:
                lane_invasion_penalty = 0.0

            step_penalty = -0.05

            # 7. 终点奖励
            success_reward = 0.0
            if self.goal_zone is not None:
                in_goal_region = self._is_in_goal_zone(transform.location)
            else:
                center_loc = target_loc
                twp = self.target_waypoints[i]
                forward_v = twp.transform.get_forward_vector()
                right_v = twp.transform.get_right_vector()
                rel = carla.Vector3D(
                    transform.location.x - center_loc.x,
                    transform.location.y - center_loc.y, 0.0)
                x_along = rel.x * forward_v.x + rel.y * forward_v.y
                y_lat = rel.x * right_v.x + rel.y * right_v.y
                in_goal_region = abs(x_along) <= 3.0 and abs(y_lat) <= 1.75
            if in_goal_region:
                success_reward = 100.0

            reward = (progress_reward + speed_reward + centering_reward
                      + yaw_reward + step_penalty + collision_penalty
                      + lane_invasion_penalty + success_reward)
            details = {
                "progress": progress_reward, "speed": speed_reward,
                "centering": centering_reward, "yaw": yaw_reward,
                "collision": collision_penalty, "lane_invasion": lane_invasion_penalty,
                "step": step_penalty, "success": success_reward,
            }
            rewards[i] = reward
            reward_details[i] = details
            reward_diagnostics[i] = _empty_diagnostics(
                progress_m=progress,
                speed_mps=speed,
                lateral_distance_m=lateral_dist,
                yaw_error_rad=angle_diff,
                collision_event=float(collision_penalty != 0.0),
                lane_invasion_sensor_event=float(
                    sensor_invasion and not crosswalk
                ),
                success_event=float(success_reward != 0.0),
            )

        # 对抗智能体处理（v1 版）
        for adv_id in self.adversarial_agent_ids:
            if adv_id in rewards:
                base_details = reward_details[adv_id]
                base_reward_v1 = (
                    base_details["progress"] * 0.3 + base_details["speed"] * 0.3
                    + base_details["centering"] * 0.2 + base_details["yaw"] * 0.1
                    + base_details["collision"] * 1.0 + base_details["lane_invasion"] * 0.5
                    + base_details["step"] * 0.1 + base_details["success"] * 0.3
                )
                adv_details_v1 = {
                    "base_driving": base_reward_v1, "cutting_in": 0.0, "blocking": 0.0,
                }
                reward_details[adv_id]["base_reward"] = rewards[adv_id]
                reward_details[adv_id].update(adv_details_v1)
                rewards[adv_id] = base_reward_v1

        self.last_reward_diagnostics = reward_diagnostics
        return rewards, reward_details

    def _compute_rewards_v2(self) -> Tuple[Dict[int, float], Dict[int, Dict]]:
        """优化版 v2 — 全连续梯度

        设计原则:
          - 所有形状分量提供从好到坏的平滑梯度，消除二元阶梯
          - 主前进信号 = Toromanoff desired-speed 期望速度带（progress 已停用，PROGRESS_WEIGHT=0），
            方向由 centering/yaw 维持；与离终点距离无关 → 不存在远车停滞问题
          - 安全惩罚仅在风险出现时激活，形成可感知的梯度
        """
        rewards = {}
        reward_details = {}
        reward_diagnostics = {}

        def _empty_diagnostics(**overrides):
            values = {
                "progress_m": 0.0,
                "distance_to_goal_m": 0.0,
                "speed_mps": 0.0,
                "closing_speed_mps": 0.0,
                "lateral_distance_m": 0.0,
                "yaw_error_rad": 0.0,
                "motion_gate": 0.0,
                "lidar_min_normalized": 1.0,
                "lidar_trigger_ray_count": 0.0,
                "ttc_candidate_count": 0.0,
                "ttc_trigger_count": 0.0,
                "ttc_min_seconds_capped": 3.0,
                "collision_event": 0.0,
                "lane_invasion_sensor_event": 0.0,
                "lane_deviation_penalty": 0.0,
                "success_event": 0.0,
            }
            values.update({key: float(value) for key, value in overrides.items()})
            return values

        # 可调超参数
        # ─── Phase B2 (R4) 重平衡 ───
        # 旧权重让 30 步撞车的 return ≈ +245 > 0，agent 学到"轰油门往前撞"。
        # 目标：让"撞车 return < 0 < 正常行驶 return < 成功 return"。
        # 新规模（按 30 步 episode 估算）:
        #   progress ≈ 0.8*5*0.72 ≈ 2.9/step → 30步 ≈ +85
        #   collision = -300                  → 撞车 return ≈ -215（强负信号）
        #   safety_lidar 翻 3 倍 → 接近碰撞前就给出连续负梯度
        # ─── progress 项（Toromanoff 改造：停用无界 progress，避免"越快越赚"压过期望速度） ───
        # 设 0 即关闭；>0 可回退为"progress + desired-speed"混合（建议同时给 progress 按目标速率封顶）。
        PROGRESS_WEIGHT = 0.0    # was 5.0；Toromanoff desired-speed 模式下停用
        # ─── 速度奖励（Toromanoff 2020 desired-speed：目标速度带内取峰，过快/过慢都减） ───
        # 用"期望速度带"取代无界 progress 作为主前进信号：带内等价对待，越界两侧都减、超速转负。
        # 带 = [5, 15] km/h；车按时间偏好会自然贴上沿 ~15km/h 巡航，等价"稳定 ≤15km/h"。
        SPEED_LOW  = 5.0 / 3.6        # 舒适速度带下界 ≈1.39 m/s (5 km/h)
        SPEED_HIGH = 15.0 / 3.6       # 舒适速度带上界 ≈4.17 m/s (15 km/h)
        SPEED_PEAK = 1.0             # 带内奖励峰值（progress 已停用，这是主前进信号）
        OVERSPEED_SLOPE = 0.4       # 超上界每 m/s 的下降斜率（可转负，超速惩罚）
        SPEED_FLOOR = -1.0           # 速度奖励下限，防极端高速把该项拉爆
        MOTION_GATE_SPEED = 2.0      # 运动门控/起步判定阈值 (m/s)
        SIGMA_CENTER = 0.3
        # 居中奖励死区（平台）：|横向偏移|<CENTER_DEADZONE 内满额、无梯度。
        # 消除尖峰居中奖励"死磕精确中心"的高增益微修正激励——这正是行驶中轻微左右
        # 摆动（欠阻尼极限环）的来源。带内大致居中即可，不诱发过冲。
        CENTER_DEADZONE = 0.4
        # ─── R-4 车道保持：连续横向偏离惩罚（抑制大幅 S 形画龙/无谓变道） ───
        # 参考 Toromanoff 2020 / Stanley 代价：对偏离车道中心的横向距离连续扣分。
        LANE_DEV_WEIGHT = 0.5      # 横向偏离惩罚斜率（每米）
        # 死区对齐到 ≥ CENTER_DEADZONE，保证内带（|d|<0.4）真正平坦、无回正梯度，
        # 不再制造微修正激励；只在明显偏离（>0.5m）时才施加回正惩罚。
        LANE_DEV_DEADZONE = 0.5    # was 0.3
        JERK_WEIGHT = 0.05
        STEER_RATE_WEIGHT = 0.05   # was 0.02：加大转向变化率惩罚，直接压制来回打方向的 S 形
        # Phase B2 修正：safety_lidar 是 11 条 ray 求和，max=-11/step 会淹没所有学习信号
        # Town02 窄路 + 前后车时几乎所有 ray 都触发。回退到原值 0.3，max=-3.3/step
        SAFETY_LIDAR_WEIGHT = 0.3  # 原值（上一轮误判改到 1.0 导致 update 7 出现 -4711 return）
        SAFE_THRESHOLD = 0.5
        # 破冻结(R5)：撞车 -300 ≫ 成功 +100 的强非对称把探索压成"原地不动"。
        # 缩小撞车惩罚（减小惩罚 ≠ 增加惩罚），仍保持 撞车 < 正常行驶 < 成功 的排序。
        COLLISION_PENALTY = -150.0  # was -300.0
        SUCCESS_REWARD = 100.0
        TIME_BONUS_MAX = 50.0
        TTC_WEIGHT_NEW = 2.0        # 回退到原值（上一轮 5.0 与 lidar 叠加每步 -5~-15）
        # 破冻结(R5)：连续"接近终点"正向势能奖励的权重（与 progress 同号，纯正向引导）
        APPROACH_WEIGHT = 0.5

        for i, vehicle in enumerate(self.vehicles):
            # 已到达终点（含 disappear-at-target 被移除的车）：不再计算奖励，给 0。
            # 必须在 not-alive 判定之前，否则到达后被移除的车会落入 not-alive 分支被误罚 -150。
            if self.reached_target.get(i, False):
                rewards[i] = 0.0
                reward_details[i] = {
                    "progress": 0.0, "approach": 0.0, "speed": 0.0, "alive": 0.0, "centering": 0.0, "yaw": 0.0,
                    "jerk": 0.0, "steer_rate": 0.0, "safety_lidar": 0.0, "ttc": 0.0,
                    "collision": 0.0, "lane_invasion": 0.0,
                    "step": 0.0, "success": 0.0, "time_bonus": 0.0,
                }
                reward_diagnostics[i] = _empty_diagnostics()
                continue

            if not vehicle.is_alive:
                rewards[i] = COLLISION_PENALTY
                reward_details[i] = {
                    "progress": 0.0, "approach": 0.0, "speed": 0.0, "alive": 0.0, "centering": 0.0, "yaw": 0.0,
                    "jerk": 0.0, "steer_rate": 0.0, "safety_lidar": 0.0, "ttc": 0.0,
                    "collision": COLLISION_PENALTY, "lane_invasion": 0.0,
                    "step": 0.0, "success": 0.0, "time_bonus": 0.0,
                }
                reward_diagnostics[i] = _empty_diagnostics(collision_event=1.0)
                continue

            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            target_loc = self._get_target_location(i)
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            # 朝终点的"闭合速度"（速度在 指向终点 单位向量上的投影，clamp≥0）。
            # 摆动根因修复：progress 已停用后 speed_reward 是主前进信号，若用总速度模长，
            # 左右摆动产生的横向速度也会被当成"速度"刷满 speed_reward + motion_gate，
            # 不接近终点也能拿满分 → 这正是画龙/摆动的激励来源。
            # 用"朝终点闭合速度"取代总速度模长：
            #   - 横向靠近终点 / 拐弯朝终点开 → 闭合速度为正 → 照常奖励（不误伤、不抑制转向）；
            #   - 左右摆动 / 垂直于终点方向乱动 → 闭合速度≈0 → 不刷分（堵住 farm）。
            _gdx = target_loc.x - transform.location.x
            _gdy = target_loc.y - transform.location.y
            _gnorm = math.hypot(_gdx, _gdy)
            if _gnorm > 1e-6:
                closing_speed = max(0.0, (velocity.x * _gdx + velocity.y * _gdy) / _gnorm)
            else:
                closing_speed = speed

            # ─── 1. 效率加权的进度奖励 ───
            # progress=delta，站着不动整项=0 → 无 farm 风险
            # 远距收敛性修复：原 proximity_factor 让"越靠近目标每米越值钱"
            #   (近车 ×2.5，远车 ×1.0)，等于结构性奖励近车、冷落远车 ——
            #   远 spawn 的车同样往前开 1 米只拿近车 40% 的 progress 奖励，
            #   叠加 success 被 γ 折没 + step 重罚，远车学不到冲终点。
            # 现把 PROXIMITY_BOOST 置 0 → progress 对远近一视同仁（干净势能塑形）。
            # 保留变量便于回调；如需"临门加速"可设小正值（建议 ≤0.3）。
            current_dist = transform.location.distance(target_loc)
            prev_dist = self.prev_distances.get(i, current_dist)
            progress = prev_dist - current_dist

            # 破冻结(R5)：去掉 efficiency 门控。原 (0.3 + 0.7*efficiency) 让低速前进
            #   只拿 30% 的 progress 奖励，"胆怯地往前挪"几乎无正反馈，助推原地不动。
            #   现改为满额，让任何前进都立即拿到正比于位移的正奖励，使"移动"严格优于"停住"。
            progress_reward = progress * PROGRESS_WEIGHT
            self.prev_distances[i] = current_dist

            # ─── 1b. "接近终点"势能塑形（R-1 修复：改为 PBRS 势能差分，farm-proof） ───
            # 旧实现按当前位置给 level 奖励 (APPROACH_WEIGHT * ratio) 且每步发放、未受
            # motion_gate 约束 → 停在终点区外可每步刷 +0.47，累计可压过 success(+100)，
            # 诱导"到终点不进、在边上刷分"。
            # 改为势能塑形 F = γΦ(s') − Φ(s)（Ng et al. 1999，保证最优策略不变）：
            #   Φ(s) = APPROACH_WEIGHT * approach_ratio(s)
            #   停车 → Φ 不变 → F = Φ·(γ−1) < 0（微负），farm 消失；
            #   向终点移动 → F > 0；远离 → F < 0。
            d0 = self.initial_distances.get(i, max(current_dist, 1.0))
            ratio_prev = max(0.0, min(1.0, (d0 - prev_dist) / d0))
            ratio_curr = max(0.0, min(1.0, (d0 - current_dist) / d0))
            approach_reward = APPROACH_WEIGHT * (self.reward_gamma * ratio_curr - ratio_prev)

            # ─── 2. 速度奖励（Toromanoff 2020 desired-speed：期望速度带，越界两侧都减） ───
            # 取代无界 progress 作为主前进信号（progress 已停用，见 PROGRESS_WEIGHT=0）：
            #   speed=0            → 0           （静止不刷分，保留防停滞起步梯度）
            #   0 → SPEED_LOW      → 线性升到峰值  （鼓励起步进入舒适带）
            #   [SPEED_LOW, HIGH]  → 峰值 plateau  （5~15 km/h 舒适带，等价对待）
            #   > SPEED_HIGH       → 线性下降转负  （超速惩罚），下限 SPEED_FLOOR
            # 零点（净惩罚起点）≈ SPEED_HIGH + SPEED_PEAK/OVERSPEED_SLOPE ≈ 6.67 m/s ≈ 24 km/h。
            # 用"朝终点闭合速度"评估：摆动的横向速度不再刷分，横向接近终点仍计入
            if closing_speed < SPEED_LOW:
                speed_reward = SPEED_PEAK * (closing_speed / SPEED_LOW)
            elif closing_speed <= SPEED_HIGH:
                speed_reward = SPEED_PEAK
            else:
                speed_reward = SPEED_PEAK - OVERSPEED_SLOPE * (closing_speed - SPEED_HIGH)
            speed_reward = max(SPEED_FLOOR, speed_reward)

            # alive_bonus 已并入速度奖励的起步线性段；保留 0 占位兼容 details["alive"] 日志。
            alive_bonus = 0.0

            # ─── 运动门控（修"原地不动 farm 奖励"致命 bug） ───
            # centering(+0.5) 和 yaw(+0.15) 原本每步无条件发放：一辆停在车道中央、
            # 车头对准目标的车，每步 farm 正奖励 → 策略收敛到"原地不动"。
            # 把这两项乘上运动门控：停车 → gate=0 → 姿态奖励=0；正常行驶 → gate=1。
            # 门控也用闭合速度：摆动(闭合≈0)→gate≈0→不发放居中/航向；接近终点→gate=1 正常
            motion_gate = min(1.0, closing_speed / MOTION_GATE_SPEED)

            # ─── 3. 车道居中奖励（带死区的平台，抑制行驶中轻微左右摆动） ───
            # 仍按最近车道中心算横向偏移（不锁定具体车道，不限制变道自由）。
            # 关键改动：用"死区平台"取代尖峰指数——|偏移|<CENTER_DEADZONE 内满额且无梯度，
            #   带外才指数衰减。尖峰只奖励"精确压中心"，最优策略=高增益横向修正，在离散时间+
            #   延迟下欠阻尼过冲 → 轻微极限环摆动；平台让"大致居中"即满额，消除微修正激励。
            current_waypoint = self.map.get_waypoint(transform.location)
            centering_reward = 0.0
            lateral_dist = 0.0   # 到最近车道中心的横向距离，供居中奖励 + R-4 偏离惩罚共用
            if current_waypoint:
                veh_loc = transform.location
                wp_loc = current_waypoint.transform.location
                vec_wp_to_veh = carla.Vector3D(
                    veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0
                )
                right_vector = current_waypoint.transform.get_right_vector()
                lateral_dist = abs(
                    vec_wp_to_veh.x * right_vector.x + vec_wp_to_veh.y * right_vector.y
                )
                d_center = max(0.0, lateral_dist - CENTER_DEADZONE)  # 死区内 = 0 → 满额平台
                centering_reward = 0.5 * math.exp(
                    -(d_center ** 2) / (2 * SIGMA_CENTER ** 2)
                )
            # 运动门控：停车时不发放居中奖励，杜绝原地 farm
            centering_reward *= motion_gate

            # ─── 4. 连续航向奖励（余弦相似度） ───
            dx = target_loc.x - transform.location.x
            dy = target_loc.y - transform.location.y
            target_yaw = math.atan2(dy, dx)
            vehicle_yaw = math.radians(transform.rotation.yaw)
            angle_diff = abs(target_yaw - vehicle_yaw)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            yaw_reward = 0.15 * (math.cos(angle_diff) + 1.0) / 2.0
            # 运动门控：停车时不发放航向奖励，杜绝原地 farm
            yaw_reward *= motion_gate

            # ─── 5. 驾驶平滑性惩罚（jerk + steer rate） ───
            jerk_penalty = 0.0
            steer_penalty = 0.0
            prev_ctrl = self.prev_controls.get(i, None)
            if prev_ctrl is not None:
                ctrl = vehicle.get_control()
                jerk_penalty = -JERK_WEIGHT * abs(
                    ctrl.throttle - prev_ctrl['throttle']
                ) / self.timestep
                steer_penalty = -STEER_RATE_WEIGHT * abs(
                    ctrl.steer - prev_ctrl['steer']
                ) / self.timestep

            # ─── 6. LiDAR 安全距离惩罚 ───
            lidar_dists = self._get_ray_detection(
                vehicle, ray_length=4.0, num_rays=11
            )
            safety_penalty = 0.0
            lidar_trigger_ray_count = 0
            for d in lidar_dists:
                if d < SAFE_THRESHOLD:
                    lidar_trigger_ray_count += 1
                    safety_penalty += (
                        -SAFETY_LIDAR_WEIGHT * (SAFE_THRESHOLD - d) / SAFE_THRESHOLD
                    )

            # ─── 7. TTC 前向碰撞风险惩罚 ───
            TTC_WEIGHT = TTC_WEIGHT_NEW  # Phase B2: was 2.0 → 5.0
            TTC_THRESHOLD = 3.0     # TTC 阈值（秒），低于此值开始惩罚
            TTC_TAU = 1.0           # 指数衰减常数
            TTC_LATERAL_RANGE = 2.0 # 同车道判定宽度（米）
            TTC_MAX_RANGE = 10.0    # 最大前向检测距离（米）

            ttc_penalty = 0.0
            ttc_candidate_count = 0
            ttc_trigger_count = 0
            ttc_min_seconds_capped = TTC_THRESHOLD
            rel_positions = self._get_relative_position_to_agents(i)
            for other_id, rel in rel_positions.items():
                # 只关心同车道正前方的车辆
                if abs(rel['lateral_dist']) < TTC_LATERAL_RANGE and rel['longitudinal_dist'] > 0:
                    if rel['longitudinal_dist'] < TTC_MAX_RANGE:
                        ttc_candidate_count += 1
                        other_vehicle = self.vehicles[other_id]
                        other_vel = other_vehicle.get_velocity()
                        other_speed = math.sqrt(
                            other_vel.x**2 + other_vel.y**2 + other_vel.z**2
                        )
                        rel_speed = speed - other_speed  # 正值表示自车更快
                        if rel_speed > 0:
                            ttc = rel['longitudinal_dist'] / rel_speed
                            ttc_min_seconds_capped = min(
                                ttc_min_seconds_capped, ttc
                            )
                            if ttc < TTC_THRESHOLD:
                                ttc_trigger_count += 1
                                ttc_penalty += -TTC_WEIGHT * math.exp(-ttc / TTC_TAU)

            # ─── 8. 碰撞惩罚（COLLISION_PENALTY，当前 -150，见上方常量定义与调参注释） ───
            crosswalk = self.crosswalk_invasion_events.get(i, False)
            collision_penalty = COLLISION_PENALTY if (
                self.collision_events.get(i, False) or crosswalk
            ) else 0.0

            # ─── 9. 车道纪律：离散压线惩罚 + 连续横向偏离惩罚（R-4） ───
            # (a) 离散压线：传感器在跨越车道线那一刻触发，按线型分级重罚（硬约束）。
            sensor_invasion = self.lane_invasion_events.get(i, False)
            lane_invasion_penalty = 0.0
            if not crosswalk and sensor_invasion:
                details_inv = getattr(self, 'lane_invasion_details', {}).get(i, {})
                marking_types = details_inv.get('marking_types', [])
                # 加重跨线惩罚（按用户要求）：强化车道纪律，压制压线/越界/上路沿。
                severity_map = {
                    'Broken': -3.0,   # 虚线：轻微 (was -1.0)
                    'Solid':  -12.0,  # 实线：中等 (was -5.0)
                    'Curb':   -25.0,  # 路沿：严重 (was -10.0)
                    'NONE':   0.0,    # 人行横道，已在 collision 中处理
                }
                if marking_types:
                    lane_invasion_penalty = sum(
                        severity_map.get(t, -12.0) for t in marking_types
                    )
                else:
                    lane_invasion_penalty = -12.0

            # (b) R-4 连续横向偏离惩罚：偏离车道中心超过死区即线性扣分（每步、不依赖压线
            #     传感器），在"还没压到线"之前就给出回正梯度，抑制 S 形画龙/无谓变道。
            #     与居中奖励合在一起形成围绕车道中心的"势能阱"。
            lane_dev_penalty = -LANE_DEV_WEIGHT * max(0.0, lateral_dist - LANE_DEV_DEADZONE)
            lane_invasion_penalty += lane_dev_penalty

            # ─── 10. 步数惩罚 ───
            # 修复历程：-0.05 太轻（timeout=−50）不足以惩罚"原地不动"；
            #   曾调到 -0.5（timeout=−500）确保"不动"比"撞车"更差。
            # 远距收敛性修复：-0.5 对远 spawn 的车是"距离重税"——它要多跑几百步，
            #   光 step 惩罚就累计 −200~−400，而被 γ 折没的 +100 又拿不到，
            #   于是远车的最优策略变成慢慢挪/不冲终点。降到 -0.1（timeout=−100），
            #   仍满足 撞车(−300) < 不动(−100) < 正常行驶 的排序，但不再过罚长途。
            step_penalty = -0.1

            # ─── 11. 终点奖励 + 时间效率加成 ───
            success_reward = 0.0
            time_bonus = 0.0

            if self.goal_zone is not None:
                in_goal_region = self._is_in_goal_zone(transform.location)
            else:
                center_loc = target_loc
                twp = self.target_waypoints[i]
                forward_v = twp.transform.get_forward_vector()
                right_v = twp.transform.get_right_vector()
                rel = carla.Vector3D(
                    transform.location.x - center_loc.x,
                    transform.location.y - center_loc.y,
                    0.0,
                )
                x_along = rel.x * forward_v.x + rel.y * forward_v.y
                y_lat = rel.x * right_v.x + rel.y * right_v.y
                in_goal_region = abs(x_along) <= 3.0 and abs(y_lat) <= 1.75

            if in_goal_region:
                success_reward = SUCCESS_REWARD
                time_bonus = max(
                    0.0, (self.max_steps - self.current_step) / self.max_steps
                ) * TIME_BONUS_MAX

            # ─── 汇总 ───
            reward = (
                progress_reward
                + approach_reward
                + speed_reward
                + alive_bonus
                + centering_reward
                + yaw_reward
                + jerk_penalty
                + steer_penalty
                + safety_penalty
                + ttc_penalty
                + collision_penalty
                + lane_invasion_penalty
                + step_penalty
                + success_reward
                + time_bonus
            )

            details = {
                "progress": progress_reward,
                "approach": approach_reward,
                "speed": speed_reward,
                "alive": alive_bonus,
                "centering": centering_reward,
                "yaw": yaw_reward,
                "jerk": jerk_penalty,
                "steer_rate": steer_penalty,
                "safety_lidar": safety_penalty,
                "ttc": ttc_penalty,
                "collision": collision_penalty,
                "lane_invasion": lane_invasion_penalty,
                "step": step_penalty,
                "success": success_reward,
                "time_bonus": time_bonus,
            }

            rewards[i] = reward
            reward_details[i] = details
            reward_diagnostics[i] = _empty_diagnostics(
                progress_m=progress,
                distance_to_goal_m=current_dist,
                speed_mps=speed,
                closing_speed_mps=closing_speed,
                lateral_distance_m=lateral_dist,
                yaw_error_rad=angle_diff,
                motion_gate=motion_gate,
                lidar_min_normalized=float(np.min(lidar_dists)),
                lidar_trigger_ray_count=lidar_trigger_ray_count,
                ttc_candidate_count=ttc_candidate_count,
                ttc_trigger_count=ttc_trigger_count,
                ttc_min_seconds_capped=ttc_min_seconds_capped,
                collision_event=float(collision_penalty != 0.0),
                lane_invasion_sensor_event=float(
                    sensor_invasion and not crosswalk
                ),
                lane_deviation_penalty=lane_dev_penalty,
                success_event=float(success_reward != 0.0),
            )

        # === 对抗智能体特殊处理 ===
        for adv_id in self.adversarial_agent_ids:
            if adv_id in rewards:
                base_reward = rewards[adv_id]
                base_details = reward_details[adv_id]
                adv_reward, adv_details = self._compute_adversarial_reward(
                    adv_id, base_details
                )
                reward_details[adv_id]["base_reward"] = base_reward
                reward_details[adv_id].update(adv_details)
                rewards[adv_id] = adv_reward

        self.last_reward_diagnostics = reward_diagnostics
        return rewards, reward_details
    
    def _check_dones(self) -> Dict[int, bool]:
        """检查是否完成。

        同时维护 self.last_terminated / self.last_truncated（修复 #2）：
          - terminated[i]: agent-specific 任务终止 → value 不 bootstrap
          - truncated[i]:  agent-specific 任务仍可继续，但被外部边界打断 → value 仍需 bootstrap
          - active_mask[i]: 下一步是否仍允许动作与 transition 采样
          - episode_end_reason: running / cooperative_success / all_agents_reached /
                                cooperative_collision / adversary_collision / max_steps /
                                all_agents_terminated
        训练侧据此在 GAE 中只对 terminated 切断价值 bootstrap，对 truncated 保留。
        """
        dones = {}
        terminated: Dict[int, bool] = {}

        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                dones[i] = True
                terminated[i] = True
                continue
            
            # 检查是否到达目标
            transform = vehicle.get_transform()
            is_collided = self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False)

            # 检查是否到达终点区域（矩形判定）
            if self.goal_zone is not None:
                in_goal_region = self._is_in_goal_zone(transform.location)
            else:
                target_wp = self.target_waypoints[i]
                center_loc = target_wp.transform.location
                forward_v = target_wp.transform.get_forward_vector()
                right_v = target_wp.transform.get_right_vector()
                rel = carla.Vector3D(
                    transform.location.x - center_loc.x,
                    transform.location.y - center_loc.y,
                    0.0,
                )
                x_along = rel.x * forward_v.x + rel.y * forward_v.y
                y_lat = rel.x * right_v.x + rel.y * right_v.y
                in_goal_region = abs(x_along) <= 3.0 and abs(y_lat) <= 1.75

            # 修复：done 对 reached_target sticky
            # 原版只有 in_goal_region 时才 done=True；若车辆进 zone 后被惯性滑出，
            # 下一帧 in_goal_region=False、无碰撞、未超时 → dones[i] 又变 False，
            # train 脚本因 all(dones) != True 而不 break，造成"到点了但回合不停"
            already_reached = self.reached_target.get(i, False)
            if in_goal_region or already_reached:
                if not already_reached:
                    # disappear-at-target（MAPF）：到达即从场景移除该车，避免它停在终点
                    # 物理阻挡后到的车，导致"会车被判碰撞"。原 ignore_vehicles_percentage
                    # 是 TrafficManager 设置，对 RL 手动控制的车无效，故改为直接移除。
                    self._despawn_reached_vehicle(i)
                self.reached_target[i] = True
                dones[i] = True
                terminated[i] = True   # 到达终点：真正终止
            elif is_collided:
                dones[i] = True
                terminated[i] = True   # 自身严重碰撞：真正终止
            elif self.current_step >= self.max_steps:
                dones[i] = True
                terminated[i] = False  # 超时截断：value 仍需 bootstrap
            else:
                dones[i] = False
                terminated[i] = False


            # # 如果到达终点，标记为已完成，并设置车辆为无碰撞模式
            # if distance < 1.0:
            #     if not self.reached_target.get(i, False):
            #         # 第一次到达终点，设置车辆忽略与其他车辆的碰撞
            #         # 这样其他车辆可以穿过它到达终点
            #         self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
            #         print(f"Agent {i} 到达终点，已设置为无碰撞模式")
            #     self.reached_target[i] = True
            #     dones[i] = True
            # elif self.current_step >= self.max_steps or is_collided:
            #     dones[i] = True
            # else:
            #     dones[i] = False
        
        # 严重碰撞（碰撞或压人行横道）的全局终止规则。
        # adversary_collision_ends_episode=False 时，蓝车只结束自身生命周期；红车结局仍由
        # 红车成功、红车碰撞或 max_steps 决定。
        def _is_severe(i: int) -> bool:
            return ((self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False))
                    and not self.reached_target.get(i, False))

        cooperative_ids = [
            i for i in range(self.num_agents) if i not in self.adversarial_agent_ids
        ]
        # 纯合作环境以全体 agent 为结局主体；对抗环境只以红车结局为全局结局。
        outcome_ids = cooperative_ids or list(range(self.num_agents))

        if self.adversary_collision_ends_episode:
            trigger_ids = list(range(self.num_agents))
        else:
            trigger_ids = cooperative_ids

        severe_ids = [i for i in trigger_ids if _is_severe(i)]
        all_outcome_reached = bool(outcome_ids) and all(
            self.reached_target.get(i, False) for i in outcome_ids
        )

        episode_end_reason = "running"
        if severe_ids:
            if any(i in cooperative_ids for i in severe_ids):
                episode_end_reason = "cooperative_collision"
                print("检测到红车严重碰撞，结束本回合")
            else:
                episode_end_reason = "adversary_collision"
                print("检测到蓝车严重碰撞，结束本回合")
        elif all_outcome_reached:
            episode_end_reason = (
                "all_agents_reached"
                if all(self.reached_target.get(i, False) for i in range(self.num_agents))
                else "cooperative_success"
            )
            print("红车已全部到达终点，结束本回合")
        elif self.current_step >= self.max_steps:
            episode_end_reason = "max_steps"
        elif dones and all(dones.values()):
            # 兜底纯对抗/自定义 agent 编组，避免没有 active agent 时空转到 max_steps。
            episode_end_reason = "all_agents_terminated"

        # agent-centric 结局语义：
        # - 红车任务按车辆分别计回报。一辆红车碰撞时，其他红车只是被场景重置打断，
        #   因而保持 terminated=False，随后记为 truncated=True。
        # - 蓝车的对抗任务以红车成功/碰撞为结局。无论红车最终成功还是碰撞，
        #   本轮对抗都已经产生结果，因此 cooperative_success/cooperative_collision
        #   对所有蓝车均为 terminated=True，不允许 bootstrap。
        # - adversary_collision_ends_episode=True 是当前入口未启用的兼容分支，本轮保持
        #   旧语义：直接碰撞蓝车 terminated，其他被连带关闭的车辆 truncated。
        if episode_end_reason in {"cooperative_success", "cooperative_collision"}:
            for i in self.adversarial_agent_ids:
                terminated[i] = True

        # 全局 episode 边界关闭所有仍存活的个体；其 terminated/truncated 原因由上面的
        # agent-specific 任务语义决定。
        if episode_end_reason != "running":
            for i in range(self.num_agents):
                dones[i] = True

        self.episode_end_reason = episode_end_reason
        self.last_terminated = {i: bool(terminated.get(i, False)) for i in range(self.num_agents)}
        self.last_truncated = {
            i: bool(dones.get(i, False) and not terminated.get(i, False))
            for i in range(self.num_agents)
        }
        for i in range(self.num_agents):
            if self.last_terminated[i] and self.last_truncated[i]:
                raise RuntimeError(f"Agent {i} cannot be both terminated and truncated")
            if bool(dones.get(i, False)) != (
                self.last_terminated[i] or self.last_truncated[i]
            ):
                raise RuntimeError(
                    f"Agent {i} lifecycle mismatch: done={dones.get(i, False)}, "
                    f"terminated={self.last_terminated[i]}, "
                    f"truncated={self.last_truncated[i]}"
                )
        self.active_mask = {
            i: bool(
                not dones.get(i, False)
                and i < len(self.vehicles)
                and self.vehicles[i] is not None
                and self.vehicles[i].is_alive
                and not self.reached_target.get(i, False)
            )
            for i in range(self.num_agents)
        }

        return dones
    
    def _get_relative_position_to_agents(self, ego_id: int) -> Dict[int, Dict[str, float]]:
        """
        计算自车相对于其他智能体的位置关系
        返回: {other_agent_id: {'lateral_dist': 横向距离, 'longitudinal_dist': 纵向距离,
                               'is_left': 是否在左侧, 'is_behind': 是否在后方}}
        """
        relative_positions = {}
        ego_vehicle = self.vehicles[ego_id]
        if not ego_vehicle.is_alive:
            return relative_positions

        ego_transform = ego_vehicle.get_transform()
        ego_loc = ego_transform.location
        ego_yaw = math.radians(ego_transform.rotation.yaw)

        # 前向和右向向量
        forward = np.array([math.cos(ego_yaw), math.sin(ego_yaw)])
        right = np.array([-math.sin(ego_yaw), math.cos(ego_yaw)])

        for other_id, other_vehicle in enumerate(self.vehicles):
            if other_id == ego_id or not other_vehicle.is_alive:
                continue

            other_loc = other_vehicle.get_location()
            rel_vec = np.array([other_loc.x - ego_loc.x, other_loc.y - ego_loc.y])

            # 投影到前向和侧向
            longitudinal_dist = float(np.dot(rel_vec, forward))  # 正值：其他车在前方
            lateral_dist = float(np.dot(rel_vec, right))          # 正值：其他车在右侧

            relative_positions[other_id] = {
                'lateral_dist': lateral_dist,
                'longitudinal_dist': longitudinal_dist,
                'is_left': lateral_dist < 0,
                'is_behind': longitudinal_dist < 0,
                'distance': math.sqrt(longitudinal_dist**2 + lateral_dist**2)
            }

        return relative_positions

    def _despawn_reached_vehicle(self, agent_id: int):
        """disappear-at-target：车辆到达终点后从场景移除，使其不再物理阻挡后到的车。

        先停掉并销毁其传感器（避免销毁瞬间触发回调写脏 collision/lane 标志），再销毁车辆本体。
        其余逻辑（obs 归零、terminated=True、reward=0）由各处的 is_alive / reached_target 守卫处理。
        全部用 try/except + is_alive 守卫，幂等且不会因重复调用报错。
        """
        # 传感器：停止监听 + 销毁
        for sensor_map in (self.collision_sensors, self.lane_sensors, self.camera_sensors):
            sensor = sensor_map.get(agent_id)
            if sensor is not None:
                try:
                    if sensor.is_listening:
                        sensor.stop()
                except Exception:
                    pass
                try:
                    if sensor.is_alive:
                        sensor.destroy()
                except Exception:
                    pass
        # 清零该车的事件标志，避免移除瞬间的残留触发碰撞/偏离判定
        self.collision_events[agent_id] = False
        self.lane_invasion_events[agent_id] = False
        self.crosswalk_invasion_events[agent_id] = False

        # 车辆本体
        try:
            vehicle = self.vehicles[agent_id]
            if vehicle is not None and vehicle.is_alive:
                vehicle.destroy()
        except Exception:
            pass

    def _compute_adversarial_reward(self, agent_id: int, base_reward_details: Dict) -> Tuple[float, Dict]:
        """
        计算对抗智能体的专用奖励
        组合: 基础驾驶奖励 + 对抗干扰奖励 + 侧面切入奖励
        """
        # 1. 基础驾驶奖励（权重调低，保留但不主导）
        base_reward = (
            base_reward_details["progress"] * 0.3
            + base_reward_details["speed"] * 0.3
            + base_reward_details["centering"] * 0.2
            + base_reward_details["yaw"] * 0.1
            + base_reward_details.get("jerk", 0.0) * 0.3
            + base_reward_details.get("steer_rate", 0.0) * 0.3
            + base_reward_details.get("safety_lidar", 0.0) * 0.3
            + base_reward_details.get("ttc", 0.0) * 0.5
            + base_reward_details["collision"] * 1.0
            + base_reward_details["lane_invasion"] * 0.5
            + base_reward_details["step"] * 0.1
            + base_reward_details["success"] * 0.3
            + base_reward_details.get("time_bonus", 0.0) * 0.3
        )

        # 2. 获取与其他智能体的相对位置
        relative_positions = self._get_relative_position_to_agents(agent_id)

        cutting_in_reward = 0.0
        blocking_reward = 0.0

        for other_id, rel in relative_positions.items():
            # 侧面切入奖励：在其他车辆侧面近距离行驶
            # 目标：横向距离小，纵向距离接近（并排或略前）
            lateral_abs = abs(rel['lateral_dist'])
            long_abs = abs(rel['longitudinal_dist'])

            # 理想切入位置：侧面 1-3米，纵向在 -5~+10米范围内
            if 1.0 < lateral_abs < 3.0 and -5.0 < rel['longitudinal_dist'] < 10.0:
                # 根据距离远近给予奖励，越近奖励越高（但不小于安全距离）
                proximity_score = 1.0 - (lateral_abs - 1.0) / 2.0  # 1米时=1.0, 3米时=0.0
                cutting_in_reward += proximity_score * 0.5

            # 阻挡奖励：在其他车辆正前方近距离
            if lateral_abs < 1.5 and 0.0 < rel['longitudinal_dist'] < 8.0:
                blocking_score = 1.0 - rel['longitudinal_dist'] / 8.0
                blocking_reward += blocking_score * 0.3

        # 3. 最终对抗奖励 = 基础驾驶 + 切入/阻挡
        adversarial_details = {
            "base_driving": base_reward,
            "cutting_in": cutting_in_reward,
            "blocking": blocking_reward,
        }

        total_adversarial_reward = base_reward + cutting_in_reward + blocking_reward

        return total_adversarial_reward, adversarial_details

    def _check_collisions(self) -> Dict[int, bool]:
        """检查碰撞（读取传感器回调设置的碰撞标志）"""
        collisions = {}
        for i in range(self.num_agents):
            has_collision = self.collision_events.get(i, False)
            has_crosswalk = self.crosswalk_invasion_events.get(i, False)
            collisions[i] = has_collision or has_crosswalk
        return collisions
    
    def get_global_state(self) -> np.ndarray:
        """获取全局状态：拼接所有智能体的完整观测（19维/agent），用于集中式Critic"""
        obs = self._get_observations()
        features = [obs[i] for i in range(self.num_agents)]
        return np.concatenate(features)

    def get_adversarial_global_state(self) -> np.ndarray:
        """获取对抗团队全局状态：拼接对抗智能体的完整观测（19维/agent），用于对抗团队集中式Critic"""
        obs = self._get_observations()
        features = [obs[aid] for aid in self.adversarial_agent_ids]
        return np.concatenate(features) if features else np.array([], dtype=np.float32)

    def load_random_scenario(self, scenario_dir: Optional[str] = None,
                            scenario_files: Optional[List[str]] = None) -> Optional[Dict]:
        """
        从场景目录随机加载一个场景配置

        Args:
            scenario_dir: 场景文件目录，默认是 data/training_scenarios
            scenario_files: 可选的场景文件列表，从中随机选择

        Returns:
            场景配置字典，如果加载失败返回 None
        """
        import json
        import random

        if scenario_dir is None:
            scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'data', 'training_scenarios')

        if not os.path.exists(scenario_dir):
            # print(f"场景目录不存在: {scenario_dir}")
            return None

        if scenario_files is None:
            scenario_files = [f for f in os.listdir(scenario_dir) if f.endswith('.json')]

        if not scenario_files:
            # print("未找到场景文件")
            return None

        random_scenario_file = random.choice(scenario_files)
        scenario_path = os.path.join(scenario_dir, random_scenario_file)

        try:
            with open(scenario_path, 'r', encoding='utf-8') as f:
                scenario_config = json.load(f)
            # print(f"已加载场景: {scenario_config.get('description', random_scenario_file)}")
            return scenario_config
        except Exception as e:
            # print(f"加载场景失败 {scenario_path}: {e}")
            return None

    def load_scenario_by_name(self, scenario_name: str,
                            scenario_dir: Optional[str] = None) -> Optional[Dict]:
        """
        按名称加载指定场景

        Args:
            scenario_name: 场景文件名（如 'scenario_01_basic.json'）
            scenario_dir: 场景目录

        Returns:
            场景配置字典
        """
        import json

        if scenario_dir is None:
            scenario_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      'data', 'training_scenarios')

        scenario_path = os.path.join(scenario_dir, scenario_name)

        if not os.path.exists(scenario_path):
            # print(f"场景文件不存在: {scenario_path}")
            return None

        try:
            with open(scenario_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            # print(f"加载场景失败: {e}")
            return None

    def set_scenario(self, scenario_config: Dict):
        """
        应用场景配置

        Args:
            scenario_config: 从 JSON 加载的场景配置字典
        """
        self.apply_scenario_config(scenario_config)

    def reset_with_random_scenario(self, scenario_dir: Optional[str] = None,
                                scenario_files: Optional[List[str]] = None) -> Tuple[Dict[int, np.ndarray], Dict]:
        """
        随机加载场景并重置环境

        Returns:
            observations, info
        """
        scenario_config = self.load_random_scenario(scenario_dir, scenario_files)
        if scenario_config is not None:
            self.set_scenario(scenario_config)
        return self.reset()

    def close(self):
        """关闭环境"""
        # 关闭TensorBoard writer
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
            print(f"TensorBoard日志已保存到: {self.tb_log_dir}")
        self._cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)

