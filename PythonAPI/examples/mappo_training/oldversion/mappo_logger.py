"""
MAPPO训练日志记录模块
使用TensorBoard记录训练过程中的各种指标
"""

import os
from typing import Dict, Optional
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

class MAPPOLogger:
    """MAPPO训练日志记录器"""
    
    def __init__(
        self,
        log_dir: str = './logs/mappo_training',
        comment: str = '',
        add_timestamp: bool = True  
    ):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志保存目录
            comment: 日志注释（会添加到目录名）
        """


        if add_timestamp:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # 如果有注释，加在时间戳后面，否则只用时间戳
            run_name = f"{current_time}_{comment}" if comment else current_time
            self.log_dir = os.path.join(log_dir, run_name)
        else:
            self.log_dir = os.path.join(log_dir, comment) if comment else log_dir

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建TensorBoard writer
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        
        # 记录计数器
        self.step_count = 0
        self.episode_count = 0
        
        # 用于记录episode级别的指标
        self.episode_rewards = {}  # {agent_id: [rewards]}
        self.episode_lengths = []
        self.episode_losses = {}  # {agent_id: {'actor_loss': [], 'critic_loss': [], 'entropy': []}}
    
    def log_step(
        self,
        rewards: Dict[int, float],
        observations: Optional[Dict[int, np.ndarray]] = None,
        actions: Optional[Dict[int, np.ndarray]] = None,
        values: Optional[Dict[int, float]] = None
    ):
        """
        记录单步指标
        
        Args:
            rewards: 每个智能体的奖励
            observations: 每个智能体的观测（可选）
            actions: 每个智能体的动作（可选）
            values: 每个智能体的价值估计（可选）
        """
        for agent_id, reward in rewards.items():
            # [修改] 移动到 Environment 分组
            self.writer.add_scalar(f'Agent_{agent_id}/Environment/Step_Reward', reward, self.step_count)
            
            # 记录观测统计
            if observations is not None and agent_id in observations:
                obs = observations[agent_id]
                # [修改] 移动到 Observation 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Distance_To_Target', obs[2] * 100, self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Speed', obs[4] * 30, self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Angle_Diff', abs(obs[3]) * 180, self.step_count)
            
            # 记录动作统计
            if actions is not None and agent_id in actions:
                action = actions[agent_id]
                # [修改] 移动到 Action 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Throttle', action[0], self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Steer', action[1], self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Brake', action[2], self.step_count)
            
            # 记录价值估计
            if values is not None and agent_id in values:
                # [修改] 移动到 Training 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Training/Value_Estimate', values[agent_id], self.step_count)
        
        self.step_count += 1
    
    def log_episode(
        self,
        episode_rewards: Dict[int, float],
        episode_length: int,
        losses: Optional[Dict[int, Dict[str, float]]] = None,
        success: Optional[Dict[int, bool]] = None
    ):
        """
        记录episode级别的指标
        
        Args:
            episode_rewards: 每个智能体的episode总奖励
            episode_length: episode长度（步数）
            losses: 每个智能体的损失（可选）
            success: 每个智能体是否成功（可选）
        """
        for agent_id, reward in rewards.items():
            # [修改] 移动到 Environment 分组
            self.writer.add_scalar(f'Agent_{agent_id}/Environment/Step_Reward', reward, self.step_count)
            
            # 记录观测统计
            if observations is not None and agent_id in observations:
                obs = observations[agent_id]
                # [修改] 移动到 Observation 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Distance_To_Target', obs[2] * 100, self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Speed', obs[4] * 30, self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Observation/Angle_Diff', abs(obs[3]) * 180, self.step_count)
            
            # 记录动作统计
            if actions is not None and agent_id in actions:
                action = actions[agent_id]
                # [修改] 移动到 Action 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Throttle', action[0], self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Steer', action[1], self.step_count)
                self.writer.add_scalar(f'Agent_{agent_id}/Action/Brake', action[2], self.step_count)
            
            # 记录价值估计
            if values is not None and agent_id in values:
                # [修改] 移动到 Training 分组
                self.writer.add_scalar(f'Agent_{agent_id}/Training/Value_Estimate', values[agent_id], self.step_count)
        
        self.step_count += 1
    
    def log_episode(
        self,
        episode_rewards: Dict[int, float],
        episode_length: int,
        losses: Optional[Dict[int, Dict[str, float]]] = None,
        success: Optional[Dict[int, bool]] = None
    ):
        """
        记录episode级别的指标
        [优化] 统一分组命名
        """
        self.episode_count += 1
        self.episode_lengths.append(episode_length)
        
        # 记录每个智能体的episode奖励
        for agent_id, reward in episode_rewards.items():
            if agent_id not in self.episode_rewards:
                self.episode_rewards[agent_id] = []
            self.episode_rewards[agent_id].append(reward)
            
            # [修改] 奖励统计分组
            self.writer.add_scalar(f'Agent_{agent_id}/Reward/Total', reward, self.episode_count)
            
            # 记录平均奖励（滑动窗口）
            if len(self.episode_rewards[agent_id]) >= 10:
                avg_reward = np.mean(self.episode_rewards[agent_id][-10:])
                self.writer.add_scalar(f'Agent_{agent_id}/Reward/Avg_10', avg_reward, self.episode_count)
            
            if len(self.episode_rewards[agent_id]) >= 100:
                avg_reward = np.mean(self.episode_rewards[agent_id][-100:])
                self.writer.add_scalar(f'Agent_{agent_id}/Reward/Avg_100', avg_reward, self.episode_count)
        
        # [修改] 全局统计分组 Global
        self.writer.add_scalar('Global/Episode_Length', episode_length, self.episode_count)
        
        if len(self.episode_lengths) >= 10:
            avg_length = np.mean(self.episode_lengths[-10:])
            self.writer.add_scalar('Global/Episode_Length_Avg_10', avg_length, self.episode_count)
        
        # 记录损失
        if losses is not None:
            for agent_id, agent_losses in losses.items():
                # [修改] 将所有损失归类到 Training 分组
                if 'actor_loss' in agent_losses:
                    self.writer.add_scalar(f'Agent_{agent_id}/Training/Loss_Actor', agent_losses['actor_loss'], self.episode_count)
                
                if 'critic_loss' in agent_losses:
                    self.writer.add_scalar(f'Agent_{agent_id}/Training/Loss_Critic', agent_losses['critic_loss'], self.episode_count)
                
                if 'entropy' in agent_losses:
                    self.writer.add_scalar(f'Agent_{agent_id}/Training/Entropy', agent_losses['entropy'], self.episode_count)
        
        # 记录成功率
        if success is not None:
            for agent_id, is_success in success.items():
                self.writer.add_scalar(
                    f'Agent_{agent_id}/Environment/Success',
                    1.0 if is_success else 0.0,
                    self.episode_count
                )
            
            success_rate = sum(success.values()) / len(success) if success else 0.0
            self.writer.add_scalar('Global/Success_Rate', success_rate, self.episode_count)
    
    def log_hyperparameters(
        self,
        hyperparameters: Dict[str, any],
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        记录超参数
        
        Args:
            hyperparameters: 超参数字典
            metrics: 指标字典（可选）
        """
        if metrics is None:
            metrics = {}
        self.writer.add_hparams(hyperparameters, metrics)
    
    def log_histogram(
        self,
        tag: str,
        values: np.ndarray,
        step: Optional[int] = None
    ):
        """
        记录直方图
        
        Args:
            tag: 标签
            values: 值数组
            step: 步数（如果为None，使用当前step_count）
        """
        if step is None:
            step = self.step_count
        self.writer.add_histogram(tag, values, step)
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: Optional[int] = None
    ):
        """
        记录标量值
        
        Args:
            tag: 标签
            value: 值
            step: 步数（如果为None，使用当前step_count）
        """
        if step is None:
            step = self.step_count
        self.writer.add_scalar(tag, value, step)
    
    def flush(self):
        """刷新写入缓冲区"""
        self.writer.flush()
    
    def close(self):
        """关闭日志记录器"""
        self.writer.close()
    
    def get_summary_stats(self) -> Dict[str, any]:
        """
        获取汇总统计信息
        
        Returns:
            统计信息字典
        """
        stats = {
            'total_steps': self.step_count,
            'total_episodes': self.episode_count,
        }
        
        # 每个智能体的统计
        for agent_id in self.episode_rewards.keys():
            if len(self.episode_rewards[agent_id]) > 0:
                stats[f'agent_{agent_id}_avg_reward'] = np.mean(self.episode_rewards[agent_id])
                stats[f'agent_{agent_id}_max_reward'] = np.max(self.episode_rewards[agent_id])
                stats[f'agent_{agent_id}_min_reward'] = np.min(self.episode_rewards[agent_id])
        
        if len(self.episode_lengths) > 0:
            stats['avg_episode_length'] = np.mean(self.episode_lengths)
            stats['max_episode_length'] = np.max(self.episode_lengths)
            stats['min_episode_length'] = np.min(self.episode_lengths)
        
        return stats

    def log_reward_breakdown(self, reward_details: Dict[int, Dict], step: int):
        """专门记录奖励构成"""
        for agent_id, details in reward_details.items():
            for name, val in details.items():
                # 在 TensorBoard 中创建一个 Reward_Breakdown 分组
                tag = f'Agent_{agent_id}/Reward_Breakdown/{name}'
                self.writer.add_scalar(tag, val, step)