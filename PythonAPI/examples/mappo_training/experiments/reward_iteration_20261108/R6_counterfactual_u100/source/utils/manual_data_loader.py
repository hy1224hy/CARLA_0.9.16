"""
手动驾驶数据加载器
将采集的手动驾驶数据转换为RLlib可用的格式
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path


class ManualDataLoader:
    """手动驾驶数据加载器"""
    
    def __init__(self, data_dir: str):
        """
        初始化数据加载器
        
        Args:
            data_dir: 手动驾驶数据目录
        """
        self.data_dir = Path(data_dir)
        self.data_files = {}
        self.loaded_data = {}
        
        # 查找数据文件
        self._find_data_files()
    
    def _find_data_files(self):
        """查找数据文件"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 查找JSON和NPZ文件
        for file_path in self.data_dir.glob("agent_*_data.json"):
            agent_id = int(file_path.stem.split("_")[1])
            self.data_files[agent_id] = str(file_path)
        
        for file_path in self.data_dir.glob("agent_*_data.npz"):
            agent_id = int(file_path.stem.split("_")[1])
            if agent_id not in self.data_files:  # 优先使用JSON
                self.data_files[agent_id] = str(file_path)
        
        if not self.data_files:
            raise FileNotFoundError(f"在 {self.data_dir} 中未找到数据文件")
        
        print(f"找到 {len(self.data_files)} 个智能体的数据文件: {self.data_files}")
    
    def load_agent_data(self, agent_id: int) -> Dict:
        """
        加载指定智能体的数据
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            包含观察、动作、奖励等的数据字典
        """
        if agent_id not in self.data_files:
            raise ValueError(f"未找到Agent {agent_id}的数据文件")
        
        if agent_id in self.loaded_data:
            return self.loaded_data[agent_id]
        
        file_path = self.data_files[agent_id]
        
        if file_path.endswith('.json'):
            data = self._load_json(file_path)
        elif file_path.endswith('.npz'):
            data = self._load_npz(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        # 验证数据
        self._validate_data(data, agent_id)
        
        self.loaded_data[agent_id] = data
        return data
    
    def _load_json(self, file_path: str) -> Dict:
        """加载JSON格式的数据"""
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        # 转换为numpy数组格式
        all_obs = []
        all_actions = []
        all_rewards = []
        all_dones = []
        all_next_obs = []
        
        for ep in raw_data["episodes"]:
            all_obs.extend(ep["observations"])
            all_actions.extend(ep["actions"])
            all_rewards.extend(ep["rewards"])
            all_dones.extend(ep["dones"])
            all_next_obs.extend(ep.get("next_observations", []))
        
        return {
            "agent_id": raw_data["agent_id"],
            "num_episodes": raw_data["num_episodes"],
            "total_steps": len(all_obs),
            "observations": np.array(all_obs, dtype=np.float32),
            "actions": np.array(all_actions, dtype=np.float32),
            "rewards": np.array(all_rewards, dtype=np.float32),
            "dones": np.array(all_dones, dtype=bool),
            "next_observations": np.array(all_next_obs, dtype=np.float32) if all_next_obs else None
        }
    
    def _load_npz(self, file_path: str) -> Dict:
        """加载NPZ格式的数据"""
        data = np.load(file_path, allow_pickle=True)
        
        return {
            "agent_id": int(data["agent_id"]),
            "num_episodes": int(data["num_episodes"]),
            "total_steps": len(data["observations"]),
            "observations": data["observations"].astype(np.float32),
            "actions": data["actions"].astype(np.float32),
            "rewards": data["rewards"].astype(np.float32),
            "dones": data["dones"].astype(bool),
            "next_observations": data.get("next_observations", None)
        }
    
    def _validate_data(self, data: Dict, agent_id: int):
        """验证数据格式和完整性"""
        # 检查必需字段
        required_fields = ["observations", "actions", "rewards", "dones"]
        for field in required_fields:
            if field not in data:
                raise ValueError(f"数据缺少必需字段: {field}")
        
        # 检查数据维度
        obs = data["observations"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        
        if len(obs) != len(actions) or len(obs) != len(rewards) or len(obs) != len(dones):
            raise ValueError(f"数据长度不一致: obs={len(obs)}, actions={len(actions)}, "
                           f"rewards={len(rewards)}, dones={len(dones)}")
        
        # 检查观察维度（应该是19维）
        if obs.shape[1] != 19:
            raise ValueError(f"观察维度不正确: 期望19维，实际{obs.shape[1]}维")
        
        # 检查动作维度（应该是3维）
        if actions.shape[1] != 3:
            raise ValueError(f"动作维度不正确: 期望3维，实际{actions.shape[1]}维")
        
        print(f"Agent {agent_id} 数据验证通过:")
        print(f"  Episodes: {data['num_episodes']}")
        print(f"  总步数: {data['total_steps']}")
        print(f"  观察形状: {obs.shape}")
        print(f"  动作形状: {actions.shape}")
    
    def filter_low_quality_data(
        self,
        data: Dict,
        min_reward: Optional[float] = None,
        filter_collisions: bool = True,
        filter_lane_invasions: bool = True
    ) -> Dict:
        """
        过滤低质量数据
        
        Args:
            data: 数据字典
            min_reward: 最小奖励阈值（None表示不过滤）
            filter_collisions: 是否过滤碰撞数据
            filter_lane_invasions: 是否过滤车道偏离数据
            
        Returns:
            过滤后的数据
        """
        # 这里可以根据需要实现更复杂的过滤逻辑
        # 例如：检测碰撞、车道偏离等
        
        mask = np.ones(len(data["rewards"]), dtype=bool)
        
        if min_reward is not None:
            mask = mask & (data["rewards"] >= min_reward)
        
        # 过滤极端负奖励（可能是碰撞）
        if filter_collisions:
            collision_threshold = -50.0
            mask = mask & (data["rewards"] > collision_threshold)
        
        filtered_data = {
            "agent_id": data["agent_id"],
            "num_episodes": data["num_episodes"],  # 这个可能需要重新计算
            "total_steps": mask.sum(),
            "observations": data["observations"][mask],
            "actions": data["actions"][mask],
            "rewards": data["rewards"][mask],
            "dones": data["dones"][mask],
        }
        
        if data.get("next_observations") is not None:
            filtered_data["next_observations"] = data["next_observations"][mask]
        
        print(f"过滤后数据: {filtered_data['total_steps']}/{data['total_steps']} 步")
        
        return filtered_data
    
    def get_statistics(self) -> Dict:
        """获取数据统计信息"""
        stats = {}
        
        for agent_id in self.data_files.keys():
            data = self.load_agent_data(agent_id)
            stats[agent_id] = {
                "num_episodes": data["num_episodes"],
                "total_steps": data["total_steps"],
                "mean_reward": float(np.mean(data["rewards"])),
                "std_reward": float(np.std(data["rewards"])),
                "min_reward": float(np.min(data["rewards"])),
                "max_reward": float(np.max(data["rewards"])),
                "mean_episode_length": data["total_steps"] / data["num_episodes"] if data["num_episodes"] > 0 else 0
            }
        
        return stats
    
    def convert_to_rllib_format(self, agent_id: int) -> Dict:
        """
        将数据转换为RLlib可用的格式
        
        Args:
            agent_id: 智能体ID
            
        Returns:
            RLlib格式的数据字典
        """
        data = self.load_agent_data(agent_id)
        
        # RLlib期望的格式（简化版，用于预训练）
        rllib_data = {
            "observations": data["observations"],
            "actions": data["actions"],
            "rewards": data["rewards"],
            "dones": data["dones"],
            "agent_id": agent_id
        }
        
        return rllib_data
    
    def get_all_agent_ids(self) -> List[int]:
        """获取所有可用的智能体ID"""
        return list(self.data_files.keys())


def main():
    """测试数据加载器"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试手动驾驶数据加载器")
    parser.add_argument("--data-dir", type=str, default="./manual_driving_data",
                       help="数据目录")
    
    args = parser.parse_args()
    
    loader = ManualDataLoader(args.data_dir)
    
    # 显示统计信息
    stats = loader.get_statistics()
    print("\n数据统计信息:")
    for agent_id, stat in stats.items():
        print(f"\nAgent {agent_id}:")
        for key, value in stat.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
