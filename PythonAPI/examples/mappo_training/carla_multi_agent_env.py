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
        max_steps: int = 1000
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
        """
        self.host = host
        self.port = port
        self.town = town
        self.num_agents = num_agents
        self.straight_road_length = straight_road_length
        self.timestep = timestep
        self.max_steps = max_steps
        
        # 连接CARLA客户端
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        # 加载世界
        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        
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
        
        # 找到直道
        self._find_straight_road()
        
        # 生成车辆
        self._spawn_vehicles()
        
        # 交通管理器
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)


        # TensorBoard记录
        self.use_tensorboard = True  # 可以通过参数控制是否启用
        
        # 获取进程ID和worker ID（用于多进程环境）
        import os
        worker_id = os.environ.get('RAY_WORKER_ID', '0')
        process_id = os.getpid()
        
        # 使用绝对路径，并为每个worker创建独立目录
        base_log_dir = os.path.abspath('logs/carla_training_reward')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 为每个worker创建独立的日志目录
        self.tb_log_dir = os.path.join(base_log_dir, f"{timestamp}_worker_{worker_id}_pid_{process_id}")
        
        if self.use_tensorboard:
            try:
                # 确保目录存在（使用绝对路径）
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
        self.episode_step = 0
   

    def _log_to_tensorboard(self, rewards: Dict[int, float], reward_details: Dict[int, Dict], step: int):
        """
        将奖励信息记录到TensorBoard
        
        Args:
            rewards: 每个agent的总奖励
            reward_details: 每个agent的奖励明细
            step: 当前步数
        """
        if not self.use_tensorboard or self.tb_writer is None:
            return
        
        for agent_id in range(self.num_agents):
            if agent_id not in rewards or agent_id not in reward_details:
                continue
            
            total_reward = rewards[agent_id]
            details = reward_details[agent_id]
            
            # 计算每部分奖励占总奖励的比例
            # 使用绝对值来计算占比，避免负值问题
            abs_total = sum(abs(v) for v in details.values())
            
            # 记录总奖励
            self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward/Total', total_reward, step)
            
            # 记录每个奖励分量及其占比
            for key, value in details.items():
                # 记录奖励分量值
                self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward/{key.capitalize()}', value, step)
                
                # 计算并记录占比
                if abs_total > 0:
                    percentage = (abs(value) / abs_total) * 100.0
                    self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward_Percentage/{key.capitalize()}', percentage, step)
                else:
                    self.tb_writer.add_scalar(f'Agent_{agent_id}/Reward_Percentage/{key.capitalize()}', 0.0, step)
            
            # 记录奖励分量的堆叠图（可选，用于可视化）
            # 创建一个字典，包含所有奖励分量
            reward_dict = {f'{k}': v for k, v in details.items()}
            reward_dict['total'] = total_reward
            self.tb_writer.add_scalars(f'Agent_{agent_id}/Reward_Components', reward_dict, step)
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
                    print(f"Agent {agent_id} 压人行横道(NONE)，按碰撞处理")
                
                # 打印调试信息（可选，用于验证传感器是否工作）
                print(f"Agent {agent_id} 车道偏离检测到! 压过的车道线类型: {marking_types}")
            else:
                self.lane_invasion_details[agent_id] = {
                    'has_invasion': True,
                    'crossed_markings': [],
                    'marking_types': []
                }
                print(f"Agent {agent_id} 车道偏离检测到! (无详细车道线信息)")
        
        sensor.listen(_on_invasion)
        print(f"Agent {agent_id} 车道偏离传感器已挂载")

    def _find_straight_road(self):
        """在Town02中找到一条直道"""
        print("正在查找Town02中的直道...")
        
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
            print("使用默认直道位置...")
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
        print(f"找到直道，长度: {len(self.straight_waypoints) * 2.0} 米")
        
        # 设置起始和目标waypoint
        if len(self.straight_waypoints) >= self.num_agents:
            # 为每个智能体分配起始位置（沿道路分布）
            if self.num_agents == 3:
                # 特殊处理：Agent 2 生成在 Agent 0 和 Agent 1 之间
                # Agent 0: 20%
                # Agent 2: 35% (在0和1中间)
                # Agent 1: 50%
                total_wp = len(self.straight_waypoints)
                self.start_waypoints.append(self.straight_waypoints[int(total_wp * 0.20)])  # Agent 0
                self.start_waypoints.append(self.straight_waypoints[int(total_wp * 0.50)])  # Agent 1
                self.start_waypoints.append(self.straight_waypoints[int(total_wp * 0.35)])  # Agent 2
                print(f"3个智能体特殊位置分配: Agent0={int(total_wp * 0.20)}, Agent1={int(total_wp * 0.50)}, Agent2={int(total_wp * 0.35)}")
            else:
                # 默认均匀分布
                step = max(1, len(self.straight_waypoints) // (self.num_agents + 1))
                for i in range(self.num_agents):
                    idx = (i + 1) * step
                    if idx < len(self.straight_waypoints):
                        self.start_waypoints.append(self.straight_waypoints[idx])
                    else:
                        self.start_waypoints.append(self.straight_waypoints[-1])

            # 目标位置是道路的末端
            self.target_waypoints = [self.straight_waypoints[-1]] * self.num_agents
        else:
            # 如果waypoint不够，使用同一个起点
            for _ in range(self.num_agents):
                self.start_waypoints.append(self.straight_waypoints[0])
                self.target_waypoints.append(self.straight_waypoints[-1])
    
    def _spawn_vehicles(self):
        """生成车辆"""
        # 清理现有车辆
        self._cleanup()
        
        # 车辆蓝图
        vehicle_bp = self.blueprint_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('color', '255,0,0')  # 红色
        
        for i in range(self.num_agents):
            spawn_transform = self.start_waypoints[i].transform
            
            # 尝试生成车辆
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
            if vehicle is None:
                # 如果生成失败，稍微调整位置
                spawn_transform.location.z += 0.5
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
            
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
        
        print(f"车辆 {agent_id} 摄像头传感器已附加")
    
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
            print(f"警告: 解析摄像头图像失败 (Agent {agent_id}): {e}")
    
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


    def _get_ray_detection(self, vehicle: carla.Actor, ray_length: float = 10.0, num_rays: int = 11) -> np.ndarray:
        """
        模拟激光雷达探测（射线检测）
        
        Args:
            vehicle: 车辆对象
            ray_length: 射线最大探测距离（米）
            num_rays: 射线数量（覆盖前方180度）
            
        Returns:
            distances: 归一化的距离数组 (shape: [num_rays])
                       1.0 表示范围内无障碍，0.0 表示紧贴障碍
        """
        transform = vehicle.get_transform()
        start_loc = transform.location
        start_loc.z += 1.0  # 抬高起点，避免射到地面
        
        # 车辆朝向（弧度）
        yaw = math.radians(transform.rotation.yaw)
        
        # 定义射线角度范围：从左(-90度)到右(+90度)
        fov = math.pi  # 180度视野
        angles = np.linspace(-fov/2, fov/2, num_rays)
        
        distances = []
        
        for angle in angles:
            ray_yaw = yaw + angle
            end_loc = carla.Location(
                x=start_loc.x + ray_length * math.cos(ray_yaw),
                y=start_loc.y + ray_length * math.sin(ray_yaw),
                z=start_loc.z
            )
            
            # 发射射线 (World.cast_ray 返回检测到的所有物体)
            # label_links 过滤掉不需要检测的物体，这里主要检测车辆和静态物体
            detections = self.world.cast_ray(start_loc, end_loc)
            
            det_dist = ray_length
            
            if detections:
                # 找到最近的障碍物
                for detection in detections:
                    # 过滤掉自己
                    if detection.label == vehicle.id:
                        continue
                        
                    # 计算距离
                    dist = start_loc.distance(detection.location)
                    if dist < det_dist:
                        det_dist = dist
            
            # 归一化：将 0~ray_length 映射到 0~1
            # 我们可以反转一下：1.0代表很近（危险），0.0代表很远（安全），或者反之。
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
                                                 avg_reward, self.episode_step)
                        
                        # 记录episode平均奖励明细
                        if self.episode_reward_details_buffer[agent_id]:
                            avg_details = {}
                            for key in ["progress", "speed", "centering", "yaw", "collision", 
                                       "lane_invasion", "step", "success"]:
                                values = [d[key] for d in self.episode_reward_details_buffer[agent_id] 
                                         if key in d]
                                if values:
                                    avg_details[key] = np.mean(values)
                            
                            for key, value in avg_details.items():
                                self.tb_writer.add_scalar(f'Agent_{agent_id}/Episode_Reward/{key.capitalize()}', 
                                                         value, self.episode_step)
                
                # 清空缓冲区
                for agent_id in range(self.num_agents):
                    self.episode_reward_buffer[agent_id] = []
                    self.episode_reward_details_buffer[agent_id] = []
        
        self.episode_step += 1
        self.current_step = 0
        
        # 重新生成车辆
        self._spawn_vehicles()

        # 初始化上一帧的距离
        self.prev_distances = {}

        # 初始化是否到达目标
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                target_loc = self.target_waypoints[i].transform.location
                curr_loc = vehicle.get_location()
                self.prev_distances[i] = curr_loc.distance(target_loc)
                self.reached_target[i] = False
            else:
                self.prev_distances[i] = 0.0
                self.reached_target[i] = False
        
        # 获取初始观察
        observations = self._get_observations()
        info = {'step': self.current_step}
        
        return observations, info
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        """获取所有智能体的观察"""
        observations = {}
        # 设定雷达参数
        NUM_RAYS = 11      # 射线数量
        RAY_LENGTH = 10.0   # 探测距离
        
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
            target_wp = self.target_waypoints[i]
            target_location = target_wp.transform.location
            
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
            base_obs = np.array([
                dx / 100.0,  # 归一化
                dy / 100.0,
                distance / 100.0,
                angle_diff / math.pi,  # 归一化到[-1, 1]
                speed / 30.0,  # 归一化速度（假设最大30 m/s）
                throttle,
                steer,
                brake
            ], dtype=np.float32)

            # 获取雷达感知信息 
            lidar_obs = self._get_ray_detection(vehicle, ray_length=RAY_LENGTH, num_rays=NUM_RAYS)
            
            # 拼接观测向量 (8 + 11 = 19维)
            obs = np.concatenate([base_obs, lidar_obs])
            
            observations[i] = obs
        
        return observations
    
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
                # 如果已经到达终点，强制停止车辆
                control = carla.VehicleControl(
                    throttle=0.0,
                    steer=0.0,
                    brake=1.0
                )
                vehicle.apply_control(control)
        
        # 更新世界
        self.world.tick()
        self.current_step += 1
        
        # 获取新观察
        observations = self._get_observations() 
        
        # 计算奖励
        rewards, reward_details = self._compute_rewards() 

        # 累积到episode缓冲区（用于计算episode平均值）
        if self.use_tensorboard:
            for agent_id in range(self.num_agents):
                if agent_id in rewards:
                    self.episode_reward_buffer[agent_id].append(rewards[agent_id])
                if agent_id in reward_details:
                    self.episode_reward_details_buffer[agent_id].append(reward_details[agent_id])

        

        # 记录到TensorBoard
        self._log_to_tensorboard(rewards, reward_details, self.current_step)
        
        # 检查是否完成（这会更新reached_target状态）
        dones = self._check_dones()
        
        info = {
            'step': self.current_step,
            'reward_details': reward_details,
            'collisions': self._check_collisions()
        }

        for i in self.collision_events: # 重置碰撞状态，避免重复惩罚
            self.collision_events[i] = False

        for i in self.lane_invasion_events: # 重置车道偏离状态，避免重复惩罚
            self.lane_invasion_events[i] = False
        for i in self.crosswalk_invasion_events:
            self.crosswalk_invasion_events[i] = False
        
        return observations, rewards, dones, info


    def _compute_rewards(self) -> Tuple[Dict[int, float], Dict[int, Dict]]:
        """计算奖励 (优化版)"""
        rewards = {}
        reward_details = {}
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                rewards[i] = -100.0
                reward_details[i] = {
                    "progress": 0.0,
                    "speed": 0.0,
                    "centering": 0.0,
                    "yaw": 0.0,
                    "collision": -100.0,  # 车辆被销毁的惩罚
                    "lane_invasion": 0.0,
                    "step": 0.0,
                    "success": 0.0
                }
                continue

            # 如果车辆已经到达终点，不再计算奖励（返回0奖励）
            if self.reached_target.get(i, False):
                rewards[i] = 0.0
                reward_details[i] = {
                    "progress": 0.0,
                    "speed": 0.0,
                    "centering": 0.0,
                    "yaw": 0.0,
                    "collision": 0.0,
                    "lane_invasion": 0.0,
                    "step": 0.0,
                    "success": 0.0
                }
                continue
            
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            target_wp = self.target_waypoints[i]
            
            # 1. 计算距离和进度
            current_dist = transform.location.distance(target_wp.transform.location)
            prev_dist = self.prev_distances.get(i, current_dist)
            
            # [核心] 进度奖励：这一步前进了多少米 
            progress = prev_dist - current_dist
            progress_reward = progress * 20.0 
            
            # 更新上一帧距离
            self.prev_distances[i] = current_dist

            # 2. 速度奖励
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) # m/s
            speed_kmh = speed * 3.6
            
            if speed_kmh < 1.0:
                # 停车惩罚：防止它停在原地刷居中奖励
                speed_reward = -0.1
            elif speed_kmh > 25.0:
                # 超速惩罚 (假设限速25km/h)
                speed_reward = -1.0
            else:
                # 正常行驶给予微小奖励，鼓励动起来
                speed_reward = 0.1

            # 3. 车道居中奖励 这里赋值有点问题
            # 3. 车道居中奖励：只有在完全居中时给予0.5奖励，其余情况为0
            current_waypoint = self.map.get_waypoint(transform.location)
            centering_reward = 0.0
            if current_waypoint:
                veh_loc = transform.location
                wp_loc = current_waypoint.transform.location
                vec_wp_to_veh = carla.Vector3D(veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                right_vector = current_waypoint.transform.get_right_vector()
                lateral_dist = abs(vec_wp_to_veh.x * right_vector.x + vec_wp_to_veh.y * right_vector.y)
                
                # 设置完全居中的阈值（0.1米），只有横向距离小于此值才给予奖励
                CENTERING_THRESHOLD = 0.05  # 米
                if lateral_dist < CENTERING_THRESHOLD:
                    centering_reward = 0.5
                else:
                    centering_reward = 0.0

            # 4. 角度/朝向奖励
            vehicle_yaw = math.radians(transform.rotation.yaw)
            # 计算目标向量的角度
            dx = target_wp.transform.location.x - transform.location.x
            dy = target_wp.transform.location.y - transform.location.y
            target_yaw = math.atan2(dy, dx)
            
            angle_diff = abs(target_yaw - vehicle_yaw)
            if angle_diff > math.pi: angle_diff = 2 * math.pi - angle_diff
            
            # 如果朝向偏差小于 30度 (0.5弧度)，给予微小奖励
            yaw_reward = 0.0
            if angle_diff < 0.2:
                yaw_reward = 0.1
            # 5. 事件惩罚
            crosswalk = self.crosswalk_invasion_events.get(i, False)
            collision_penalty = -100.0 if (self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False)) else 0.0
            # collision_penalty = -100.0 if self.collision_events.get(i, False) else 0.0
           
            # 车道偏离惩罚：使用传感器检测；压人行横道已按碰撞处理，不再重复扣车道偏离
            sensor_invasion = self.lane_invasion_events.get(i, False)
            
            if crosswalk:
                lane_invasion_penalty = 0.0
            elif sensor_invasion:
                lane_invasion_penalty = -5.0
                details_inv = getattr(self, 'lane_invasion_details', {}).get(i, {})
                marking_types = details_inv.get('marking_types', [])
                if self.current_step % 50 == 0:
                    print(f"Agent {i} 车道偏离惩罚触发! 传感器检测: {sensor_invasion}, "
                          f"车道线类型: {marking_types}")
            else:
                lane_invasion_penalty = 0.0
            # 6. 时间步惩罚 (Living Penalty)
            # 每活一步扣一点分，迫使它尽快到达终点
            step_penalty = -0.05

            # 7. 终点奖励
            # success_reward = 0.0
            # if current_dist < 1.5: # 放宽一点判定范围
            #     success_reward = 100.0

            # 与 _check_dones 中一致
            veh_loc = transform.location
            center_loc = target_wp.transform.location
            forward = target_wp.transform.get_forward_vector()
            right = target_wp.transform.get_right_vector()

            rel = carla.Vector3D(
                veh_loc.x - center_loc.x,
                veh_loc.y - center_loc.y,
                0.0
            )

            x_along = rel.x * forward.x + rel.y * forward.y
            y_lat   = rel.x * right.x   + rel.y * right.y

            half_len = 3.0
            half_width = 1.75

            in_goal_region = (abs(x_along) <= half_len) and (abs(y_lat) <= half_width)

            success_reward = 0.0
            if in_goal_region:
                success_reward = 100.0

            # 总奖励汇总
            # 预期: 正常行驶每步约 +1.0 ~ +2.0 分
            # 停车: 每步 -0.15 分
            reward = (
                progress_reward +      # 主项
                speed_reward +         # 辅项
                centering_reward +     # 辅项
                yaw_reward +           # 辅项
                step_penalty +         # 促使快速完成
                collision_penalty +    # 事件
                lane_invasion_penalty +# 事件
                success_reward         # 稀疏大奖
            )
            
            # 封装明细
            details = {
                "progress": progress_reward,
                "speed": speed_reward,
                "centering": centering_reward,
                "yaw": yaw_reward,
                "collision": collision_penalty,
                "lane_invasion": lane_invasion_penalty,
                "step": step_penalty,
                "success": success_reward
            }

            rewards[i] = reward
            reward_details[i] = details

            # print(f"Agent {i} reward: {reward:.2f}")
            # print(f"Details: {details}")
            
            # 打印调试 (可选，训练时建议注释掉)
            # if i == 0 and self.current_step % 10 == 0:
            #     print(f"R_prog:{progress_reward:.2f} R_cen:{centering_reward:.2f} R_spd:{speed_reward:.2f}")

        return rewards, reward_details
    
    def _check_dones(self) -> Dict[int, bool]:
        """检查是否完成"""
        dones = {}
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                dones[i] = True
                continue
            
            # 检查是否到达目标
            transform = vehicle.get_transform()
            target_wp = self.target_waypoints[i]
            
            distance = transform.location.distance(target_wp.transform.location)
            # is_collided = self.collision_events.get(i, False)
            is_collided = self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False)

            # 检查是否到达终点区域
            veh_loc = transform.location
            center_loc = target_wp.transform.location

            forward = target_wp.transform.get_forward_vector()
            right = target_wp.transform.get_right_vector()

            # 相对向量
            rel = carla.Vector3D(
                veh_loc.x - center_loc.x,
                veh_loc.y - center_loc.y,
                0.0
            )

            # 在车道前向 / 横向坐标
            x_along = rel.x * forward.x + rel.y * forward.y
            y_lat   = rel.x * right.x   + rel.y * right.y

            # 终点矩形区域尺寸（你可以按需要调）
            half_len = 3.0     # 沿车道方向半长 3m，总长 6m
            half_width = 1.75  # 车道横向半宽，约等于一条车道宽的一半

            in_goal_region = (abs(x_along) <= half_len) and (abs(y_lat) <= half_width)

            if in_goal_region:
                if not self.reached_target.get(i, False):
                    self.traffic_manager.ignore_vehicles_percentage(vehicle, 100.0)
                    print(f"Agent {i} 到达终点区域，已设置为无碰撞模式")
                self.reached_target[i] = True
                dones[i] = True
            elif self.current_step >= self.max_steps or is_collided:
                dones[i] = True
            else:
                dones[i] = False


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
        
        # 如果所有智能体都到达了终点，将所有智能体的 dones 设置为 True，结束回合
        all_reached_target = all(self.reached_target.get(i, False) for i in range(self.num_agents))
        if all_reached_target:
            print("所有智能体都已到达终点，结束本回合")
            for i in range(self.num_agents):
                dones[i] = True

        # 如果任何一个智能体发生严重碰撞（碰撞或压人行横道），将所有智能体的 dones 设置为 True，结束回合
        any_severe_collision = any(
            # self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False) 
            # for i in range(self.num_agents)
            (self.collision_events.get(i, False) or self.crosswalk_invasion_events.get(i, False))
            and not self.reached_target.get(i, False)
            for i in range(self.num_agents)
        )
        if any_severe_collision:
            print("检测到严重碰撞（碰撞或压人行横道），结束本回合")
            for i in range(self.num_agents):
                dones[i] = True
        
        return dones
    
    def _check_collisions(self) -> Dict[int, bool]:
        """检查碰撞"""
        collisions = {}
        for i in range(self.num_agents):
            collisions[i] = False  # 简化版本，可以添加碰撞检测
        
        return collisions
    
    def get_global_state(self) -> np.ndarray:
        """
        获取全局状态（用于集中式critic）
        
        Returns:
            global_state: 所有智能体的全局状态
        """
        global_features = []
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                global_features.extend([0.0] * 6)
                continue
            
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            target_wp = self.target_waypoints[i]
            
            # 位置
            global_features.append(transform.location.x / 100.0)
            global_features.append(transform.location.y / 100.0)
            
            # 速度
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            global_features.append(speed / 30.0)
            
            # 朝向
            global_features.append(math.sin(math.radians(transform.rotation.yaw)))
            global_features.append(math.cos(math.radians(transform.rotation.yaw)))
            
            # 到目标的距离
            distance = transform.location.distance(target_wp.transform.location)
            global_features.append(distance / 100.0)
        
        return np.array(global_features, dtype=np.float32)
    
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

