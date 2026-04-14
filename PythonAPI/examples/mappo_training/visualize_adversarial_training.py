"""
对抗训练可视化模块
使用CARLA俯视摄像头显示训练区域，并显示实时收益
"""

import pygame
from pygame import transform
import numpy as np
import carla
import weakref
import threading
from typing import Dict, Optional


class AdversarialVisualizer:
    """对抗训练可视化器"""

    def __init__(
        self,
        host: str = '127.0.0.1',
        port: int = 2000,
        width: int = 800,
        height: int = 600,
        fps: int = 30
    ):
        """
        初始化可视化器

        Args:
            host: CARLA服务器地址
            port: CARLA服务器端口
            width: 窗口宽度
            height: 窗口高度
            fps: 帧率
        """
        pygame.init()
        pygame.font.init()
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.fps = fps

        # 创建窗口
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("对抗训练可视化 - Adversarial Training")

        # 初始化字体
        self._init_fonts()

        # 颜色定义
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'gray': (128, 128, 128),
            'dark_gray': (50, 50, 50),
        }

        # 智能体颜色和标签
        self.agent_info = {
            0: {'color': (255, 0, 0), 'label': 'Pretrained 0'},
            1: {'color': (0, 255, 0), 'label': 'Pretrained 1'},
            2: {'color': (0, 0, 255), 'label': 'Adversarial'},
        }

        # 时钟
        self.clock = pygame.time.Clock()
        self.running = True
        self.data_lock = threading.Lock()

        # CARLA连接
        self._connect_carla()

        # 俯视摄像头
        self.top_camera = None
        self.top_camera_image = None
        self._setup_top_camera()

        # 收益信息
        self.current_rewards = {0: 0.0, 1: 0.0, 2: 0.0}
        self.episode_rewards = {0: 0.0, 1: 0.0, 2: 0.0}

        # 帧计数器，用于定期检查车辆
        self.frame_count = 0

    def _init_fonts(self):
        """初始化字体"""
        import os
        import sys

        font_paths = []

        if sys.platform == 'win32':
            font_paths = [
                'C:/Windows/Fonts/simhei.ttf',
                'C:/Windows/Fonts/simsun.ttc',
                'C:/Windows/Fonts/msyh.ttc',
            ]
        elif sys.platform == 'linux':
            font_paths = [
                '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
                '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
            ]
        elif sys.platform == 'darwin':
            font_paths = [
                '/System/Library/Fonts/PingFang.ttc',
                '/System/Library/Fonts/STHeiti Light.ttc',
            ]

        font_file = None
        for path in font_paths:
            if os.path.exists(path):
                font_file = path
                break

        if font_file:
            try:
                self.font_small = pygame.font.Font(font_file, 18)
                self.font_medium = pygame.font.Font(font_file, 24)
                self.font_large = pygame.font.Font(font_file, 32)
                print(f"使用字体: {font_file}")
            except Exception:
                self.font_small = pygame.font.Font(None, 18)
                self.font_medium = pygame.font.Font(None, 24)
                self.font_large = pygame.font.Font(None, 32)
        else:
            self.font_small = pygame.font.Font(None, 18)
            self.font_medium = pygame.font.Font(None, 24)
            self.font_large = pygame.font.Font(None, 32)

    def _connect_carla(self):
        """连接CARLA服务器"""
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        print(f"已连接CARLA服务器: {self.host}:{self.port}")

    def _setup_top_camera(self):
        """设置俯视摄像头"""
        try:
            # 获取车辆位置作为摄像头参考点
            vehicles = self.world.get_actors().filter('vehicle.*')
            if len(vehicles) == 0:
                print("警告: 未找到车辆，等待车辆生成...")
                return

            # 使用第一辆车作为参考
            ref_vehicle = vehicles[2]
            ref_transform = ref_vehicle.get_transform()
            ref_loc = ref_transform.location

            # 在高空俯视位置创建摄像头
            cam_location = carla.Location(
                x=ref_loc.x + 10,
                y=ref_loc.y,
                z=ref_loc.z + 50  # 高度50米俯视
            )

            cam_transform = carla.Transform(
                location=cam_location,
                rotation=carla.Rotation(pitch=-90, yaw=0, roll=0)  # 垂直向下
            )

            # 摄像头蓝图
            bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '800')
            bp.set_attribute('image_size_y', '600')
            bp.set_attribute('fov', '90')

            # 生成摄像头
            self.top_camera = self.world.spawn_actor(bp, cam_transform)

            # 设置回调
            weak_self = weakref.ref(self)
            self.top_camera.listen(lambda image: self._parse_image(weak_self, image))

            print("俯视摄像头已设置")

        except Exception as e:
            print(f"设置俯视摄像头失败: {e}")
            import traceback
            traceback.print_exc()

    def _parse_image(self, weak_self, image):
        """解析摄像头图像"""
        self = weak_self()
        if not self:
            return

        try:
            # 使用Raw格式获取原始数据
            image.convert(carla.ColorConverter.Raw)
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))

            # CARLA图像是BGRA格式，需要转换为RGB
            array = array[:, :, [2, 1, 0]]  # BGR -> RGB

            with self.data_lock:
                self.top_camera_image = array
        except Exception as e:
            print(f"图像解析错误: {e}")

    def update_rewards(self, current_rewards: Dict[int, float] = None,
                     episode_rewards: Dict[int, float] = None):
        """更新收益信息"""
        try:
            # 使用非阻塞锁
            if self.data_lock.acquire(blocking=False):
                if current_rewards:
                    self.current_rewards.update(current_rewards)
                if episode_rewards:
                    self.episode_rewards.update(episode_rewards)
                self.data_lock.release()
        except Exception:
            pass

    def _draw_text(self, text: str, x: int, y: int, color: tuple,
                  font: Optional[pygame.font.Font] = None, align: str = 'left'):
        """绘制文本"""
        if font is None:
            font = self.font_small

        text_surface = font.render(str(text), True, color)
        text_rect = text_surface.get_rect()

        if align == 'center':
            text_rect.center = (x, y)
        elif align == 'right':
            text_rect.right = x
            text_rect.top = y
        else:
            text_rect.left = x
            text_rect.top = y

        self.screen.blit(text_surface, text_rect)

    def _draw_reward_panel(self, agent_id: int, x: int, y: int, width: int = 200):
        """绘制收益面板"""
        color = self.agent_info[agent_id]['color']
        label = self.agent_info[agent_id]['label']

        # 背景
        panel_rect = pygame.Rect(x, y, width, 90)
        pygame.draw.rect(self.screen, self.colors['dark_gray'], panel_rect)
        pygame.draw.rect(self.screen, color, panel_rect, 2)

        # 标题
        self._draw_text(label, x + width // 2, y + 12, color, self.font_medium, 'center')

        # 当前步收益
        curr_reward = self.current_rewards.get(agent_id, 0.0)
        curr_color = self.colors['green'] if curr_reward >= 0 else self.colors['red']
        self._draw_text(f"Step: {curr_reward:.2f}", x + width // 2, y + 40,
                      curr_color, self.font_small, 'center')

        # Episode总收益
        ep_reward = self.episode_rewards.get(agent_id, 0.0)
        ep_color = self.colors['green'] if ep_reward >= 0 else self.colors['red']
        self._draw_text(f"Episode: {ep_reward:.2f}", x + width // 2, y + 62,
                      ep_color, self.font_small, 'center')

    def render(self):
        """渲染一帧"""
        # 清屏
        self.screen.fill(self.colors['black'])

        # 显示摄像头图像
        with self.data_lock:
            if self.top_camera_image is not None:
                try:
                    image_surface = pygame.surfarray.make_surface(self.top_camera_image.swapaxes(0, 1))
                    image_surface = transform.scale(image_surface, (self.width, self.height - 150))
                    self.screen.blit(image_surface, (0, 0))
                except Exception:
                    # 如果图像显示失败，显示占位
                    self._draw_text("Camera Signal Lost", self.width // 2, self.height // 2,
                                  self.colors['gray'], self.font_medium, 'center')
            else:
                self._draw_text("Waiting for Camera...", self.width // 2, self.height // 2,
                              self.colors['gray'], self.font_medium, 'center')

        # 绘制收益面板
        panel_width = 200
        panel_y = self.height - 140
        panel_spacing = 220
        start_x = (self.width - (3 * panel_spacing - 20)) // 2

        for agent_id in [0, 1, 2]:
            panel_x = start_x + agent_id * panel_spacing
            self._draw_reward_panel(agent_id, panel_x, panel_y, panel_width)

        # 底部提示
        self._draw_text("Press ESC or close window to exit", self.width // 2, self.height - 20,
                      self.colors['gray'], self.font_small, 'center')

        # 更新显示
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        """运行可视化循环"""
        print("=== 对抗训练可视化已启动 ===")
        print("红色: 预训练 Agent 0")
        print("绿色: 预训练 Agent 1")
        print("蓝色: 对抗 Agent 2")
        print("按 ESC 或关闭窗口退出")

        try:
            while self.running:
                try:
                    # 处理事件
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                self.running = False

                    # 定期检查车辆并设置摄像头（每60帧约2秒检查一次）
                    self.frame_count += 1
                    if self.top_camera is None and self.frame_count % 60 == 0:
                        print("尝试设置俯视摄像头...")
                        self._setup_top_camera()

                    self.render()
                except Exception as e:
                    print(f"渲染循环错误: {e}")
                    import traceback
                    traceback.print_exc()

        except KeyboardInterrupt:
            print("\n可视化已停止")
        finally:
            self.close()

    def close(self):
        """关闭可视化器"""
        self.running = False

        if self.top_camera and self.top_camera.is_alive:
            self.top_camera.stop()
            self.top_camera.destroy()

        pygame.quit()
        print("可视化窗口已关闭")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="对抗训练可视化")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=750)
    parser.add_argument("--fps", type=int, default=30)

    args = parser.parse_args()

    try:
        visualizer = AdversarialVisualizer(
            host=args.host,
            port=args.port,
            width=args.width,
            height=args.height,
            fps=args.fps
        )
        visualizer.run()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
