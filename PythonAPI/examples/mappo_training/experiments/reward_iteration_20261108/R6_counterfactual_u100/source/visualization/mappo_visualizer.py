"""
MAPPO训练可视化模块
使用pygame显示训练过程中的车辆观测和状态信息
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import pygame
import numpy as np
from typing import Dict, List, Optional
import math


class MAPPOVisualizer:
    """MAPPO训练可视化器"""
    
    def __init__(
        self,
        width: int = 1200,
        height: int = 800,
        fps: int = 30
    ):
        """
        初始化可视化器
        
        Args:
            width: 窗口宽度
            height: 窗口高度
            fps: 帧率
        """
        pygame.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.fps = fps
        
        # 创建窗口
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MAPPO Training Visualization")
        
        # 初始化字体（支持中文）
        self._init_fonts()
        
        # 颜色定义
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'orange': (255, 165, 0),
            'gray': (128, 128, 128),
            'dark_gray': (64, 64, 64),
        }
        
        # 车辆颜色（每个智能体不同颜色）
        self.vehicle_colors = [
            (255, 0, 0),    # 红色 - 智能体0
            (0, 0, 255),    # 蓝色 - 智能体1
            (0, 255, 0),    # 绿色 - 智能体2
            (255, 165, 0),  # 橙色 - 智能体3
        ]
        
        # 时钟
        self.clock = pygame.time.Clock()
        
        # 数据记录（用于绘制历史曲线）
        self.reward_history = {}  # {agent_id: [rewards]}
        self.max_history_length = 100
        
        # 是否运行
        self.running = True
    
    def _init_fonts(self):
        """初始化字体，支持中文显示"""
        import os
        import sys
        
        # 尝试使用系统中文字体
        font_paths = []
        
        # Windows系统字体路径
        if sys.platform == 'win32':
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',  # 黑体
                'C:/Windows/Fonts/simsun.ttc',   # 宋体
                'C:/Windows/Fonts/msyh.ttc',     # 微软雅黑
                'C:/Windows/Fonts/simkai.ttf',   # 楷体
            ]
        # Linux系统字体路径
        elif sys.platform == 'linux':
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
                '/usr/share/fonts/truetype/arphic/uming.ttc',
            ]
        # macOS系统字体路径
        elif sys.platform == 'darwin':
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
            ]
        
        # 查找可用的字体文件
        font_file = None
        for path in font_paths:
            if os.path.exists(path):
                font_file = path
                break
        
        # 如果找到字体文件，使用它；否则使用默认字体（可能不支持中文）
        if font_file:
            try:
                self.font_small = pygame.font.Font(font_file, 20)
                self.font_medium = pygame.font.Font(font_file, 24)
                self.font_large = pygame.font.Font(font_file, 32)
                print(f"使用字体: {font_file}")
            except Exception as e:
                print(f"警告: 无法加载字体 {font_file}: {e}")
                self.font_small = pygame.font.Font(None, 20)
                self.font_medium = pygame.font.Font(None, 24)
                self.font_large = pygame.font.Font(None, 32)
        else:
            # 使用默认字体（不支持中文，但至少能显示英文）
            print("警告: 未找到中文字体，中文可能显示为乱码")
            self.font_small = pygame.font.Font(None, 20)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)
    
    def handle_events(self):
        """处理pygame事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def draw_text(
        self,
        text: str,
        x: int,
        y: int,
        color: tuple = (255, 255, 255),
        font: Optional[pygame.font.Font] = None,
        align: str = 'left'
    ):
        """
        绘制文本
        
        Args:
            text: 文本内容
            x: x坐标
            y: y坐标
            color: 颜色
            font: 字体（如果为None，使用小字体）
            align: 对齐方式（'left', 'center', 'right'）
        """
        if font is None:
            font = self.font_small
        
        text_surface = font.render(str(text), True, color)
        text_rect = text_surface.get_rect()
        
        if align == 'center':
            text_rect.center = (x, y)
        elif align == 'right':
            text_rect.right = x
            text_rect.top = y
        else:  # left
            text_rect.left = x
            text_rect.top = y
        
        self.screen.blit(text_surface, text_rect)
        return text_rect.bottom
    
    def draw_rect(self, x: int, y: int, width: int, height: int, color: tuple, alpha: int = 255):
        """绘制半透明矩形"""
        s = pygame.Surface((width, height))
        s.set_alpha(alpha)
        s.fill(color)
        self.screen.blit(s, (x, y))
    
    def draw_bar(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        value: float,
        max_value: float,
        color: tuple,
        label: str = ""
    ):
        """
        绘制进度条
        
        Args:
            x: x坐标
            y: y坐标
            width: 宽度
            height: 高度
            value: 当前值
            max_value: 最大值
            color: 颜色
            label: 标签
        """
        # 绘制背景
        pygame.draw.rect(self.screen, self.colors['dark_gray'], (x, y, width, height))
        
        # 绘制进度
        if max_value > 0:
            fill_width = int((value / max_value) * width)
            fill_width = max(0, min(fill_width, width))
            pygame.draw.rect(self.screen, color, (x, y, fill_width, height))
        
        # 绘制边框
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, width, height), 1)
        
        # 绘制标签
        if label:
            self.draw_text(label, x, y - 20, self.colors['white'], self.font_small)
            self.draw_text(f"{value:.2f}", x + width + 5, y + height // 2 - 10, 
                          self.colors['white'], self.font_small)
    
    def draw_observation_panel(
        self,
        agent_id: int,
        obs: np.ndarray,
        x: int,
        y: int,
        width: int = 300
    ):
        """
        绘制观测信息面板
        
        Args:
            agent_id: 智能体ID
            obs: 观测向量 [dx, dy, distance, angle_diff, speed, throttle, steer, brake]
            x: x坐标
            y: y坐标
            width: 面板宽度
        """
        # 背景
        self.draw_rect(x, y, width, 280, self.colors['dark_gray'], 200)
        
        # 标题
        y_offset = y + 10
        self.draw_text(f"智能体 {agent_id} 观测信息", x + width // 2, y_offset, 
                      self.vehicle_colors[agent_id % len(self.vehicle_colors)], 
                      self.font_medium, 'center')
        y_offset += 30
        
        # 观测值标签
        obs_labels = [
            ("相对位置 X", obs[0] * 100, "m"),
            ("相对位置 Y", obs[1] * 100, "m"),
            ("到目标距离", obs[2] * 100, "m"),
            ("角度差", obs[3] * 180, "°"),
            ("速度", obs[4] * 30, "m/s"),
            ("油门", obs[5], ""),
            ("转向", obs[6], ""),
            ("刹车", obs[7], ""),
        ]
        
        for i, (label, value, unit) in enumerate(obs_labels):
            text = f"{label}: {value:.2f} {unit}"
            self.draw_text(text, x + 10, y_offset, self.colors['white'], self.font_small)
            y_offset += 25
            
            # 对于某些值，绘制进度条
            if i >= 5:  # throttle, steer, brake
                bar_x = x + 10
                bar_y = y_offset - 15
                bar_width = width - 20
                bar_height = 8
                
                if i == 5:  # throttle
                    self.draw_bar(bar_x, bar_y, bar_width, bar_height, value, 1.0, 
                                self.colors['green'], "")
                elif i == 6:  # steer
                    # steer在[-1, 1]，需要转换到[0, 1]用于显示
                    normalized_value = (value + 1) / 2
                    color = self.colors['yellow'] if abs(value) < 0.1 else self.colors['orange']
                    self.draw_bar(bar_x, bar_y, bar_width, bar_height, normalized_value, 1.0, 
                                color, "")
                elif i == 7:  # brake
                    self.draw_bar(bar_x, bar_y, bar_width, bar_height, value, 1.0, 
                                self.colors['red'], "")
    
    def draw_reward_panel(
        self,
        agent_id: int,
        current_reward: float,
        episode_reward: float,
        x: int,
        y: int,
        width: int = 300
    ):
        """
        绘制奖励信息面板
        
        Args:
            agent_id: 智能体ID
            current_reward: 当前步奖励
            episode_reward: 当前episode总奖励
            x: x坐标
            y: y坐标
            width: 面板宽度
        """
        # 背景
        self.draw_rect(x, y, width, 120, self.colors['dark_gray'], 200)
        
        # 标题（使用英文）
        y_offset = y + 10
        self.draw_text(f"Agent {agent_id} Reward", x + width // 2, y_offset, 
                      self.vehicle_colors[agent_id % len(self.vehicle_colors)], 
                      self.font_medium, 'center')
        y_offset += 30
        
        # 当前奖励
        self.draw_text(f"Step Reward: {current_reward:.2f}", x + 10, y_offset, 
                      self.colors['white'], self.font_small)
        y_offset += 25
        
        # Episode奖励
        self.draw_text(f"Episode Total: {episode_reward:.2f}", x + 10, y_offset, 
                      self.colors['white'], self.font_small)
        y_offset += 25
        
        # 平均奖励（如果有历史数据）
        if agent_id in self.reward_history and len(self.reward_history[agent_id]) > 0:
            avg_reward = np.mean(self.reward_history[agent_id][-10:])
            self.draw_text(f"Avg (10 steps): {avg_reward:.2f}", x + 10, y_offset, 
                          self.colors['cyan'], self.font_small)
    
    def draw_training_info(
        self,
        episode: int,
        step: int,
        max_episodes: int,
        max_steps: int,
        x: int,
        y: int
    ):
        """
        绘制训练信息
        
        Args:
            episode: 当前episode
            step: 当前步
            max_episodes: 最大episode数
            max_steps: 最大步数
            x: x坐标
            y: y坐标
        """
        # 背景
        self.draw_rect(x, y, 400, 100, self.colors['dark_gray'], 200)
        
        # 标题（使用英文）
        self.draw_text("Training Progress", x + 200, y + 10, self.colors['white'], 
                      self.font_large, 'center')
        
        y_offset = y + 40
        
        # Episode进度
        episode_progress = episode / max_episodes if max_episodes > 0 else 0
        self.draw_text(f"Episode: {episode}/{max_episodes}", x + 10, y_offset, 
                      self.colors['white'], self.font_medium)
        self.draw_bar(x + 10, y_offset + 25, 380, 15, episode_progress, 1.0, 
                     self.colors['green'], "")
        
        # Step进度
        step_progress = step / max_steps if max_steps > 0 else 0
        y_offset += 50
        self.draw_text(f"Step: {step}/{max_steps}", x + 10, y_offset, 
                      self.colors['white'], self.font_medium)
        self.draw_bar(x + 10, y_offset + 25, 380, 15, step_progress, 1.0, 
                     self.colors['blue'], "")
    
    def draw_minimap(
        self,
        vehicle_positions: Dict[int, tuple],
        target_positions: Dict[int, tuple],
        x: int,
        y: int,
        width: int = 300,
        height: int = 300
    ):
        """
        绘制小地图（车辆位置）
        
        Args:
            vehicle_positions: 车辆位置 {(agent_id): (x, y)}
            target_positions: 目标位置 {(agent_id): (x, y)}
            x: x坐标
            y: y坐标
            width: 宽度
            height: 高度
        """
        # 背景
        self.draw_rect(x, y, width, height, self.colors['black'], 200)
        pygame.draw.rect(self.screen, self.colors['white'], (x, y, width, height), 2)
        
        # 标题（使用英文）
        self.draw_text("Position Map", x + width // 2, y + 10, self.colors['white'], 
                      self.font_medium, 'center')
        
        # 计算缩放和平移（使所有车辆和目标都可见）
        if vehicle_positions or target_positions:
            all_positions = list(vehicle_positions.values()) + list(target_positions.values())
            if all_positions:
                min_x = min(p[0] for p in all_positions)
                max_x = max(p[0] for p in all_positions)
                min_y = min(p[1] for p in all_positions)
                max_y = max(p[1] for p in all_positions)
                
                # 添加边距
                range_x = max_x - min_x if max_x != min_x else 100
                range_y = max_y - min_y if max_y != min_y else 100
                min_x -= range_x * 0.1
                max_x += range_x * 0.1
                min_y -= range_y * 0.1
                max_y += range_y * 0.1
                range_x = max_x - min_x
                range_y = max_y - min_y
                
                # 缩放因子
                scale_x = (width - 40) / range_x if range_x > 0 else 1
                scale_y = (height - 40) / range_y if range_y > 0 else 1
                scale = min(scale_x, scale_y)
                
                # 绘制目标位置
                for agent_id, (tx, ty) in target_positions.items():
                    px = x + 20 + (tx - min_x) * scale
                    py = y + 40 + (ty - min_y) * scale
                    pygame.draw.circle(self.screen, self.colors['yellow'], (int(px), int(py)), 5)
                    self.draw_text(f"T{agent_id}", int(px) + 10, int(py) - 10, 
                                  self.colors['yellow'], self.font_small)
                
                # 绘制车辆位置
                for agent_id, (vx, vy) in vehicle_positions.items():
                    px = x + 20 + (vx - min_x) * scale
                    py = y + 40 + (vy - min_y) * scale
                    color = self.vehicle_colors[agent_id % len(self.vehicle_colors)]
                    pygame.draw.circle(self.screen, color, (int(px), int(py)), 8)
                    pygame.draw.circle(self.screen, self.colors['white'], (int(px), int(py)), 8, 2)
                    self.draw_text(f"A{agent_id}", int(px) + 10, int(py) - 10, 
                                  color, self.font_small)
                    
                    # 绘制到目标的连线
                    if agent_id in target_positions:
                        tx, ty = target_positions[agent_id]
                        tpx = x + 20 + (tx - min_x) * scale
                        tpy = y + 40 + (ty - min_y) * scale
                        pygame.draw.line(self.screen, color, (int(px), int(py)), 
                                       (int(tpx), int(tpy)), 1)
    
    def update_reward_history(self, agent_id: int, reward: float):
        """更新奖励历史"""
        if agent_id not in self.reward_history:
            self.reward_history[agent_id] = []
        self.reward_history[agent_id].append(reward)
        if len(self.reward_history[agent_id]) > self.max_history_length:
            self.reward_history[agent_id].pop(0)
    
    def draw_single_camera(
        self,
        agent_id: int,
        image: np.ndarray,
        x: int,
        y: int,
        width: int = 300,
        height: int = 225
    ):
        """
        绘制单个智能体的摄像头图像（新方法，用于灵活布局）
        """
        if image is None:
            # 如果没有图像，画一个占位框
            self.draw_rect(x, y, width, height, self.colors['black'])
            pygame.draw.rect(self.screen, self.colors['gray'], (x, y, width, height), 1)
            self.draw_text("No Signal", x + width//2, y + height//2, self.colors['gray'], align='center')
            return

        import pygame.transform
        # 转换图像格式 (Height, Width, Channel) -> (Width, Height, Channel)
        image_surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        scaled_image = pygame.transform.scale(image_surface, (width, height))
        
        # 绘制图像
        self.screen.blit(scaled_image, (x, y))
        
        # 绘制边框（使用智能体对应的颜色）
        color = self.vehicle_colors[agent_id % len(self.vehicle_colors)]
        pygame.draw.rect(self.screen, color, (x, y, width, height), 3)
        
        # 绘制标签背景和文字
        label_bg_height = 24
        self.draw_rect(x, y, width, label_bg_height, (0, 0, 0), 180)
        self.draw_text(
            f"Agent {agent_id} Camera",
            x + width // 2,
            y + 2,
            color,
            self.font_small,
            'center'
        )

    def render(
        self,
        observations: Dict[int, np.ndarray],
        rewards: Dict[int, float],
        episode_rewards: Dict[int, float],
        vehicle_positions: Dict[int, tuple],
        target_positions: Dict[int, tuple],
        episode: int,
        step: int,
        max_episodes: int,
        max_steps: int,
        camera_images: Optional[Dict[int, np.ndarray]] = None
    ):
        """
        渲染一帧（优化后的布局）
        """
        # 1. 清屏
        self.screen.fill((30, 30, 30))
        
        # 布局常量定义
        margin = 20
        col_width = 320  # 左右侧栏宽度
        center_width = 400 # 中间地图宽度
        
        # 计算X坐标
        left_col_x = margin
        right_col_x = self.width - col_width - margin
        center_col_x = (self.width - center_width) // 2
        
        # ==================== 顶部区域 ====================
        # 绘制训练信息（居中显示）
        self.draw_training_info(episode, step, max_episodes, max_steps, 
                              center_col_x, 10) # y=10
        
        # ==================== 中间区域 (地图) ====================
        # 地图放在进度条下方
        map_y = 130
        self.draw_minimap(vehicle_positions, target_positions, 
                         center_col_x, map_y, 
                         center_width, center_width) # 正方形地图

        # ==================== 左右侧栏 (智能体信息) ====================
        # 定义内容的起始Y坐标
        content_start_y = 80 
        
        # 遍历两个智能体（Agent 0 放左边，Agent 1 放右边）
        # 如果有更多智能体，可以根据奇偶性分配
        for agent_id in [0, 1]:
            # 确定当前智能体的 X 坐标
            if agent_id == 0:
                curr_x = left_col_x
            else:
                curr_x = right_col_x
            
            curr_y = content_start_y
            
            # 1. 绘制摄像头 (放在最上方，视觉冲击力最强)
            if camera_images and agent_id in camera_images:
                self.draw_single_camera(agent_id, camera_images[agent_id], curr_x, curr_y, col_width, 240)
                curr_y += 250 # 图片高度 + 间距
            else:
                # 占位
                self.draw_rect(curr_x, curr_y, col_width, 240, self.colors['black'])
                self.draw_text("Waiting for Camera...", curr_x + col_width//2, curr_y + 120, self.colors['gray'], align='center')
                curr_y += 250

            # 2. 绘制观测面板
            if agent_id in observations:
                self.draw_observation_panel(agent_id, observations[agent_id], curr_x, curr_y, col_width)
                curr_y += 290 # 面板高度 + 间距
            
            # 3. 绘制奖励面板
            current_reward = rewards.get(agent_id, 0.0)
            ep_reward = episode_rewards.get(agent_id, 0.0)
            self.update_reward_history(agent_id, current_reward)
            
            self.draw_reward_panel(agent_id, current_reward, ep_reward, curr_x, curr_y, col_width)

        
        instructions = [
            "Press ESC to close visualization | Red: Agent 0 | Blue: Agent 1",
        ]
        y_pos = self.height - 30
        for instruction in instructions:
            self.draw_text(instruction, self.width // 2, y_pos, 
                          self.colors['gray'], self.font_small, 'center')
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(self.fps)
    def close(self):
        """关闭可视化器"""
        pygame.quit()

