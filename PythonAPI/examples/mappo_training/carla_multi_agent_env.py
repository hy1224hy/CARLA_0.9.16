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


        # 状态信息
        self.current_step = 0
        self.straight_waypoints: List[carla.Waypoint] = []
        self.start_waypoints: List[carla.Waypoint] = []
        self.target_waypoints: List[carla.Waypoint] = []
        
        # 找到直道
        self._find_straight_road()
        
        # 生成车辆
        self._spawn_vehicles()
        
        # 交通管理器
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)

    # def _get_local_location(self, vehicle, target_location):
    #     """
    #     将世界坐标系的点转换为车辆局部坐标系
    #     返回: (forward_dist, lateral_dist)
    #     """
    #     v_trans = vehicle.get_transform()
    #     v_loc = v_trans.location
        
    #     # 1. 计算世界坐标系下的相对偏移量
    #     d_vec = [
    #         target_location.x - v_loc.x,
    #         target_location.y - v_loc.y,
    #         target_location.z - v_loc.z
    #     ]
        
    #     # 2. 获取车辆的旋转矩阵
    #     rad_yaw = math.radians(v_trans.rotation.yaw)
    #     cos_y = math.cos(rad_yaw)
    #     sin_y = math.sin(rad_yaw)
        
    #     # 3. 旋转变换
    #     forward_dist = d_vec[0] * cos_y + d_vec[1] * sin_y
    #     lateral_dist = d_vec[1] * cos_y - d_vec[0] * sin_y
        
    #     return forward_dist, lateral_dist

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
        
        # 回调函数
        def _on_invasion(event):
            # 记录发生了偏离
            self.lane_invasion_events[agent_id] = True
            # 可以打印是压了实线还是虚线: event.crossed_lane_markings
            
        sensor.listen(_on_invasion)

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
        self.current_step = 0
        
        # 重新生成车辆
        self._spawn_vehicles()

        # 初始化上一帧的距离
        self.prev_distances = {}
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                target_loc = self.target_waypoints[i].transform.location
                curr_loc = vehicle.get_location()
                self.prev_distances[i] = curr_loc.distance(target_loc)
            else:
                self.prev_distances[i] = 0.0
        
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
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive and i in actions:
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
        
        # 更新世界
        self.world.tick()
        self.current_step += 1
        
        # 获取新观察
        observations = self._get_observations()
        
        # 计算奖励
        rewards, reward_details = self._compute_rewards() 
        
        # 检查是否完成
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
        
        return observations, rewards, dones, info
    
    # def _compute_rewards(self) -> Dict[int, float]:
    #     """计算奖励"""
    #     rewards = {}
        
    #     for i, vehicle in enumerate(self.vehicles):
    #         if not vehicle.is_alive:
    #             rewards[i] = -100.0  # 车辆被销毁的惩罚
    #             continue
            
    #         transform = vehicle.get_transform()
    #         velocity = vehicle.get_velocity()
    #         target_wp = self.target_waypoints[i]
            
    #         # 当前位置
    #         current_location = transform.location
    #         target_location = target_wp.transform.location
            
    #         # 计算到目标的距离
    #         dx = target_location.x - current_location.x
    #         dy = target_location.y - current_location.y
    #         distance = math.sqrt(dx**2 + dy**2)
            
    #         # 速度
    #         speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
    #         # 奖励设计
    #         reward = 0.0
            
    #         # 距离奖励（越近越好）
    #         distance_reward = -distance / 100.0
            
    #         # 速度奖励（保持合理速度）
    #         if speed > 5.0 and speed < 15.0:  # 理想速度范围
    #             speed_reward = 5.0
    #         elif speed > 15.0:
    #             speed_reward = 5.0 - (speed - 15.0) / 10.0  # 超速惩罚
    #         else:
    #             speed_reward = speed / 5.0  # 低速奖励
            
    #         # 方向奖励（朝向目标）
    #         vehicle_yaw = math.radians(transform.rotation.yaw)
    #         target_yaw = math.atan2(dy, dx)
    #         angle_diff = abs(target_yaw - vehicle_yaw)
    #         if angle_diff > math.pi:
    #             angle_diff = 2 * math.pi - angle_diff
            
    #         direction_reward = 1.0 - (angle_diff / math.pi)  # 角度越小越好
            
    #         # 无碰撞时综合奖励
    #         reward = distance_reward + 0.5 * speed_reward + 0.3 * direction_reward

    #         # 车道偏离奖励惩罚
    #         # 如果检测到压线
    #         if self.lane_invasion_events.get(i, False):
    #             reward -= 1.0  # 给予压线惩罚
    #             print(f"Agent {i} 压线，给予惩罚。")
            
    #         # 车道居中鼓励
    #         # 获取当前位置对应的最近路点
    #         current_waypoint = self.map.get_waypoint(transform.location)
            
    #         if current_waypoint:
    #             # 计算车辆位置与路点位置的向量
    #             veh_loc = transform.location
    #             wp_loc = current_waypoint.transform.location
                
    #             # 向量: 路点 -> 车辆
    #             vec_wp_to_veh = carla.Vector3D(veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                
    #             # 获取路点的右向量 (用于计算横向偏移)
    #             right_vector = current_waypoint.transform.get_right_vector()
                
    #             # 计算点积，得到在横向方向上的投影长度 (即横向偏移距离)
    #             lateral_offset = vec_wp_to_veh.x * right_vector.x + vec_wp_to_veh.y * right_vector.y
    #             lateral_dist = abs(lateral_offset)
                
    #             # 获取车道宽度的一半
    #             lane_half_width = current_waypoint.lane_width / 2.0
                
    #             # 计算居中系数: 0 (在边缘) -> 1 (在中心)
    #             # 限制在 [0, 1] 范围内
    #             centering_factor = max(0.0, 1.0 - (lateral_dist / lane_half_width))
                
    #             # 添加到总奖励 (系数设为 0.5，根据需要调整权重)
    #             centering_reward = 5 * centering_factor
    #             # print(centering_reward)
    #             reward += centering_reward


    #         # 碰撞惩罚
    #         if self.collision_events.get(i, False):
    #             reward -= 100.0  # 给予巨大的惩罚
    #             print(f"Agent {i} 发生碰撞！给予惩罚。")
            
    #         # 如果到达目标附近
    #         if distance < 1.0:
    #             reward += 500.0  # 到达奖励
            
    #         rewards[i] = reward
        
    #     return rewards

    def _compute_rewards(self) -> Tuple[Dict[int, float], Dict[int, Dict]]:
        """计算奖励 (优化版)"""
        rewards = {}
        reward_details = {}
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                rewards[i] = -100.0
                continue
            
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            target_wp = self.target_waypoints[i]
            
            # 1. 计算距离和进度
            current_dist = transform.location.distance(target_wp.transform.location)
            prev_dist = self.prev_distances.get(i, current_dist)
            
            # [核心] 进度奖励：这一步前进了多少米 (单位: m)
            # 乘以系数，比如 10，让它成为主导奖励
            progress = prev_dist - current_dist
            progress_reward = progress * 20.0 
            
            # 更新上一帧距离
            self.prev_distances[i] = current_dist

            # 2. 速度奖励
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) # m/s
            speed_kmh = speed * 3.6
            
            if speed_kmh < 1.0:
                # [惩罚] 停车惩罚：防止它停在原地刷居中奖励
                speed_reward = -0.1
            elif speed_kmh > 25.0:
                 # [惩罚] 超速惩罚 (假设限速40km/h)
                speed_reward = -1.0
            else:
                # [鼓励] 正常行驶给予微小奖励，鼓励动起来
                speed_reward = 0.1

            # 3. 车道居中奖励 (权重降低)
            current_waypoint = self.map.get_waypoint(transform.location)
            centering_reward = 0.0
            if current_waypoint:
                veh_loc = transform.location
                wp_loc = current_waypoint.transform.location
                vec_wp_to_veh = carla.Vector3D(veh_loc.x - wp_loc.x, veh_loc.y - wp_loc.y, 0)
                right_vector = current_waypoint.transform.get_right_vector()
                lateral_dist = abs(vec_wp_to_veh.x * right_vector.x + vec_wp_to_veh.y * right_vector.y)
                lane_half_width = current_waypoint.lane_width / 2.0
                
                # 只有当偏离较小时才给奖励，避免它为了回正而剧烈偏移
                centering_factor = max(0.0, 1.0 - (lateral_dist / lane_half_width))
                # 权重从 5.0 降到 0.5，确保它小于进度奖励
                centering_reward = 0.5 * centering_factor

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
            collision_penalty = -100.0 if self.collision_events.get(i, False) else 0.0
            lane_invasion_penalty = -5.0 if self.lane_invasion_events.get(i, False) else 0.0 # 提高压线惩罚

            # 6. 时间步惩罚 (Living Penalty)
            # 每活一步扣一点分，迫使它尽快到达终点
            step_penalty = -0.05

            # 7. 终点奖励
            success_reward = 0.0
            if current_dist < 2.0: # 放宽一点判定范围
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

            print(f"Agent {i} reward: {reward:.2f}")
            print(f"Details: {details}")
            
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
            is_collided = self.collision_events.get(i, False)

            done = distance < 5.0 or self.current_step >= self.max_steps or is_collided
            dones[i] = done
           
        
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
        self._cleanup()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        

