"""
手动驾驶数据采集脚本
用于采集专家驾驶数据，作为MAPPO训练的专家示范
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import argparse
import json
import os
import time
import numpy as np
from typing import Dict, List
import pygame
from pygame.locals import K_ESCAPE, K_w, K_s, K_a, K_d, K_UP, K_DOWN, K_LEFT, K_RIGHT, K_SPACE

try:
    from environment.carla_multi_agent_env import CarlaMultiAgentEnv
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from environment.carla_multi_agent_env import CarlaMultiAgentEnv


class ManualControlRecorder:
    """手动控制数据记录器"""
    
    def __init__(
        self,
        agent_id: int,
        host: str = '127.0.0.1',
        port: int = 2000,
        town: str = 'Town02',
        output_dir: str = './manual_driving_data',
        max_steps: int = 1000
    ):
        """
        初始化手动控制记录器
        
        Args:
            agent_id: 要控制的智能体ID (0 或 1)
            host: CARLA服务器地址
            port: CARLA服务器端口
            town: 地图名称
            output_dir: 数据保存目录
            max_steps: 每个episode最大步数
        """
        self.agent_id = agent_id
        self.output_dir = output_dir
        self.max_steps = max_steps
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化环境（只生成一辆车用于手动控制）
        print(f"正在初始化CARLA环境（Agent {agent_id}）...")
        self.env = CarlaMultiAgentEnv(
            host=host,
            port=port,
            town=town,
            num_agents=1,  # 只生成一辆车
            straight_road_length=100.0,
            timestep=0.1,
            max_steps=max_steps
        )
        
        # 初始化pygame用于键盘输入
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption(f"Manual Control - Agent {agent_id}")
        self.font = pygame.font.Font(None, 36)
        self.clock = pygame.time.Clock()
        
        # 控制状态
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0

        # 平滑控制参数
        self.throttle_accel = 0.15  # 油门加速度（每帧增加量）
        self.throttle_decel = 0.1   # 油门减速度（每帧减少量）
        self.brake_accel = 0.2      # 刹车加速度
        self.brake_decel = 0.15     # 刹车减速度
        self.steer_accel = 0.003    # 转向加速度（基于时间）
        self.steer_decel = 0.05     # 转向减速度（每帧衰减率）
        self.steer_cache = 0.0      # 转向缓存值

        # 时间跟踪
        self.last_time = time.time()
        
        # 数据记录
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_observations": []
        }
        self.all_episodes = []
        
        print(f"环境初始化完成！")
        print(f"控制说明：")
        print(f"  W/↑: 油门")
        print(f"  S/↓: 刹车")
        print(f"  A/←: 左转")
        print(f"  D/→: 右转")
        print(f"  ESC: 结束当前episode并保存")
        print(f"  Q: 退出程序")
    
    def _process_keyboard(self):
        """处理键盘输入"""
        keys = pygame.key.get_pressed()
        
        # 计算时间增量（用于平滑控制）
        current_time = time.time()
        dt = current_time - self.last_time
        dt = min(dt, 0.1)  # 限制最大时间步，避免跳跃
        self.last_time = current_time
        
        # 处理油门（平滑增加/减少）
        if keys[K_w] or keys[K_UP]:
            # 按下时逐渐增加油门
            self.throttle = min(self.throttle + self.throttle_accel, 1.0)
            # 如果同时按下刹车，优先处理刹车
            if keys[K_s] or keys[K_DOWN]:
                self.throttle = max(self.throttle - self.throttle_decel * 2, 0.0)
        else:
            # 释放时逐渐减少油门
            self.throttle = max(self.throttle - self.throttle_decel, 0.0)
        
        # 处理刹车（平滑增加/减少）
        if keys[K_s] or keys[K_DOWN]:
            # 按下时逐渐增加刹车
            self.brake = min(self.brake + self.brake_accel, 1.0)
            # 如果同时按下油门，优先处理刹车
            if keys[K_w] or keys[K_UP]:
                self.throttle = max(self.throttle - self.throttle_decel * 2, 0.0)
        else:
            # 释放时逐渐减少刹车
            self.brake = max(self.brake - self.brake_decel, 0.0)
        
        # 处理转向（平滑累积/衰减）
        steer_increment = self.steer_accel * (dt * 1000)  # 转换为毫秒增量
        
        if keys[K_a] or keys[K_LEFT]:
            # 左转：如果当前是右转，先归零
            if self.steer_cache > 0:
                self.steer_cache = 0.0
            # 累积左转值
            self.steer_cache -= steer_increment
        elif keys[K_d] or keys[K_RIGHT]:
            # 右转：如果当前是左转，先归零
            if self.steer_cache < 0:
                self.steer_cache = 0.0
            # 累积右转值
            self.steer_cache += steer_increment
        else:
            # 释放时逐渐衰减转向
            if abs(self.steer_cache) < 0.01:
                self.steer_cache = 0.0
            else:
                self.steer_cache *= (1.0 - self.steer_decel)
        
        # 限制转向范围
        self.steer_cache = max(-1.0, min(1.0, self.steer_cache))
        self.steer = self.steer_cache
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            elif event.type == pygame.KEYDOWN:
                if event.key == K_ESCAPE:
                    return 'end_episode'
                elif event.key == pygame.K_q:
                    return 'quit'
        
        return 'continue'

        # # 重置控制
        # self.throttle = 0.0
        # self.steer = 0.0
        # self.brake = 0.0
        
        # # 油门
        # if keys[K_w] or keys[K_UP]:
        #     self.throttle = 1.0
        
        # # 刹车
        # if keys[K_s] or keys[K_DOWN]:
        #     self.brake = 1.0
        
        # # 转向
        # if keys[K_a] or keys[K_LEFT]:
        #     self.steer = -1.0
        # elif keys[K_d] or keys[K_RIGHT]:
        #     self.steer = 1.0
        
        # # 处理事件
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         return 'quit'
        #     elif event.type == pygame.KEYDOWN:
        #         if event.key == K_ESCAPE:
        #             return 'end_episode'
        #         elif event.key == pygame.K_q:
        #             return 'quit'
        
        # return 'continue'
    
    def _render_hud(self, step: int, reward: float, episode: int):
        """渲染HUD信息"""
        camera_images = self.env.get_camera_images()
        camera_image = camera_images.get(0, None)  # 获取agent 0的摄像头图像
        
        if camera_image is not None:
            # pygame需要 (width, height, 3)，所以需要swapaxes
            # 同时需要将RGB转为BGR（pygame.surfarray.make_surface期望BGR）
            image_surface = pygame.surfarray.make_surface(camera_image.swapaxes(0, 1))
            # 缩放图像以适应窗口（如果需要）
            display_width, display_height = 800, 600
            if image_surface.get_width() != display_width or image_surface.get_height() != display_height:
                image_surface = pygame.transform.scale(image_surface, (display_width, display_height))
            self.screen.blit(image_surface, (0, 0))
        else:
            # 如果没有摄像头图像，显示黑屏
            self.screen.fill((0, 0, 0))
        
        # 在图像上叠加HUD信息（使用半透明背景）
        y_offset = 20
        
        # 创建半透明背景用于HUD文本
        hud_surface = pygame.Surface((300, 400), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 180))  # 半透明黑色背景
        self.screen.blit(hud_surface, (10, 10))
        
        
        texts = [
            f"Agent {self.agent_id} - Episode {episode}",
            f"Step: {step}/{self.max_steps}",
            f"Reward: {reward:.2f}",
            "",
            "Controls:",
            "W/↑: Throttle",
            "S/↓: Brake",
            "A/←: Steer Left",
            "D/→: Steer Right",
            "",
            "ESC: End Episode",
            "Q: Quit"
        ]
        
        for text in texts:
            surface = self.font.render(text, True, (255, 255, 255))
            self.screen.blit(surface, (20, y_offset))
            y_offset += 30
        
        # 显示当前控制输入
        control_y = 400
        control_texts = [
            f"Throttle: {self.throttle:.2f}",
            f"Steer: {self.steer:.2f}",
            f"Brake: {self.brake:.2f}"
        ]
        for text in control_texts:
            surface = self.font.render(text, True, (0, 255, 0))
            self.screen.blit(surface, (20, control_y))
            control_y += 30
        
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS for display
    
    def record_episode(self) -> bool:
        """记录一个episode的数据"""
        print(f"\n开始记录新episode...")
        
        # 重置环境
        observations, info = self.env.reset()
        obs = observations[0]  # 只有一个智能体，ID为0
        
        # 重置episode数据
        self.current_episode = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "dones": [],
            "next_observations": []
        }

        # 重置控制状态
        self.throttle = 0.0
        self.steer = 0.0
        self.brake = 0.0
        self.steer_cache = 0.0
        self.last_time = time.time()
        
        
        step = 0
        total_reward = 0.0
        
        while step < self.max_steps:
            # 处理键盘输入
            action_result = self._process_keyboard()
            if action_result == 'quit':
                return False
            elif action_result == 'end_episode':
                print("手动结束episode")
                break
            
            # 构建动作
            action = np.array([self.throttle, self.steer, self.brake], dtype=np.float32)
            actions = {0: action}
            
            # 执行一步
            next_obs, rewards, dones, info = self.env.step(actions)
            reward = rewards[0]
            done = dones[0]
            next_obs_array = next_obs[0]
            
            # 记录数据
            self.current_episode["observations"].append(obs.tolist())
            self.current_episode["actions"].append(action.tolist())
            self.current_episode["rewards"].append(float(reward))
            self.current_episode["dones"].append(bool(done))
            self.current_episode["next_observations"].append(next_obs_array.tolist())
            
            total_reward += reward
            step += 1
            
            # 渲染HUD
            self._render_hud(step, reward, len(self.all_episodes) + 1)
            
            # 检查是否完成
            if done:
                print(f"Episode完成 (步数: {step}, 总奖励: {total_reward:.2f})")
                break
            
            obs = next_obs_array
        
        # 保存episode数据
        if len(self.current_episode["observations"]) > 0:
            self.all_episodes.append(self.current_episode.copy())
            print(f"Episode已保存 (总步数: {len(self.current_episode['observations'])}, 总奖励: {total_reward:.2f})")
            return True
        else:
            print("Episode为空，未保存")
            return False
    
    def save_data(self, format: str = 'json'):
        """保存采集的数据"""
        if len(self.all_episodes) == 0:
            print("没有数据可保存！")
            return
        
        output_data = {
            "agent_id": self.agent_id,
            "num_episodes": len(self.all_episodes),
            "total_steps": sum(len(ep["observations"]) for ep in self.all_episodes),
            "episodes": self.all_episodes
        }
        
        if format == 'json':
            filename = os.path.join(self.output_dir, f"agent_{self.agent_id}_data.json")
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n数据已保存到: {filename}")
            print(f"  Episodes: {len(self.all_episodes)}")
            print(f"  总步数: {output_data['total_steps']}")
        elif format == 'npz':
            filename = os.path.join(self.output_dir, f"agent_{self.agent_id}_data.npz")
            # 将数据转换为numpy数组
            all_obs = []
            all_actions = []
            all_rewards = []
            all_dones = []
            all_next_obs = []
            
            for ep in self.all_episodes:
                all_obs.extend(ep["observations"])
                all_actions.extend(ep["actions"])
                all_rewards.extend(ep["rewards"])
                all_dones.extend(ep["dones"])
                all_next_obs.extend(ep["next_observations"])
            
            np.savez(
                filename,
                agent_id=self.agent_id,
                observations=np.array(all_obs),
                actions=np.array(all_actions),
                rewards=np.array(all_rewards),
                dones=np.array(all_dones),
                next_observations=np.array(all_next_obs),
                num_episodes=len(self.all_episodes)
            )
            print(f"\n数据已保存到: {filename}")
            print(f"  Episodes: {len(self.all_episodes)}")
            print(f"  总步数: {len(all_obs)}")
    
    def run(self):
        """运行数据采集循环"""
        print(f"\n开始数据采集（Agent {self.agent_id}）")
        print("按任意键开始第一个episode...")
        
        episode_count = 0
        while True:
            # 等待用户开始
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        waiting = False
                    elif event.type == pygame.QUIT:
                        self.cleanup()
                        return
                
                self._render_hud(0, 0.0, episode_count)
                time.sleep(0.1)
            
            # 记录episode
            success = self.record_episode()
            if not success:
                break
            
            episode_count += 1
            
            # 询问是否继续
            print(f"\n已采集 {episode_count} 个episodes")
            print("按任意键继续下一个episode，或按Q退出...")
            
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            waiting = False
                            self.cleanup()
                            return
                        else:
                            waiting = False
                    elif event.type == pygame.QUIT:
                        self.cleanup()
                        return
                
                self._render_hud(0, 0.0, episode_count)
                time.sleep(0.1)
        
        # 保存数据
        self.save_data()
    
    def cleanup(self):
        """清理资源"""
        pygame.quit()
        self.env.close()
        print("资源已清理")


def main():
    parser = argparse.ArgumentParser(description="手动驾驶数据采集")
    parser.add_argument("--agent-id", type=int, required=True, choices=[0, 1],
                       help="要控制的智能体ID (0 或 1)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="CARLA服务器地址")
    parser.add_argument("--port", type=int, default=2000,
                       help="CARLA服务器端口")
    parser.add_argument("--town", type=str, default="Town02",
                       help="地图名称")
    parser.add_argument("--output-dir", type=str, default="./manual_driving_data",
                       help="数据保存目录")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="每个episode最大步数")
    
    args = parser.parse_args()
    
    recorder = ManualControlRecorder(
        agent_id=args.agent_id,
        host=args.host,
        port=args.port,
        town=args.town,
        output_dir=args.output_dir,
        max_steps=args.max_steps
    )
    
    try:
        recorder.run()
    except KeyboardInterrupt:
        print("\n\n中断采集，正在保存数据...")
        recorder.save_data()
        recorder.cleanup()
    except Exception as e:
        print(f"\n发生错误: {e}")
        recorder.save_data()
        recorder.cleanup()
        raise


if __name__ == "__main__":
    main()
