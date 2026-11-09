"""
PSRO 策略池管理
================
- RedTeamPool : 加载多个已训练好的 RED MAPPO ckpt，按索引提供 frozen Actor dict
- BlueTeamPool: 管理 PSRO 训练过程中产出的 BLUE 团队 ckpt（含 2 个 actor + 共享 critic）

RED 池组建方式（前置 Phase 0）：
  把不同 seed / 不同 update 步 / 不同超参的 best.pt 拷到同一目录下，
  RedTeamPool.from_dir(...) 会按文件名排序加载。

BLUE 池增长方式：
  PSRO 每轮 train_blue_oracle 结束后调 BlueTeamPool.add(...) 增加一个团队 ckpt。
"""
import os
import glob
import copy
from typing import Dict, List, Optional

import torch

# 把 mappo_training 根加进 path（被 -m training.psro.* 调用时已在 path，但独立 import 时需要）
import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

from models.actor_critic import Actor, AgentConditionedCritic


class RedTeamPool:
    """加载并管理多个 RED MAPPO ckpt。每个 ckpt 包含 cooperative actor dict。

    设计原则：
      - ckpt 在 init 时统一加载到 CPU（lazy 拷贝到 device 在 get_actors 时）
      - actor 始终 eval + requires_grad=False
      - 同一 RED 团队的多次 get_actors 返回独立副本（避免训练阶段误改）
    """

    def __init__(self, ckpt_paths: List[str], cooperative_ids: List[int],
                 obs_dim: int = 19, act_dim: int = 3, hidden_dim: int = 128,
                 device: str = 'cpu'):
        if not ckpt_paths:
            raise ValueError("RedTeamPool 至少需要 1 个 ckpt")
        self.ckpt_paths = list(ckpt_paths)
        self.cooperative_ids = list(cooperative_ids)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        self._raw_states: List[Dict[int, dict]] = []  # 每元素：{cid: state_dict (on CPU)}
        for p in self.ckpt_paths:
            if not os.path.exists(p):
                raise FileNotFoundError(f"RED ckpt 不存在: {p}")
            ckpt = torch.load(p, map_location='cpu')
            actors_state = ckpt.get("actors", {})
            for cid in self.cooperative_ids:
                if cid not in actors_state:
                    raise KeyError(f"{p} 缺少 cooperative agent {cid} 的权重")
            self._raw_states.append({cid: actors_state[cid] for cid in self.cooperative_ids})
        print(f"[RedTeamPool] 加载 {len(self._raw_states)} 个 RED 团队，"
              f"cooperative_ids={self.cooperative_ids}")

    def __len__(self) -> int:
        return len(self._raw_states)

    def __iter__(self):
        for i in range(len(self)):
            yield self.get_actors(i)

    def get_actors(self, idx: int) -> Dict[int, Actor]:
        """返回第 idx 个 RED 团队的 {cid: Actor (eval, frozen, on device)}。"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"RedTeamPool 索引越界: {idx} (size={len(self)})")
        state = self._raw_states[idx]
        actors: Dict[int, Actor] = {}
        for cid in self.cooperative_ids:
            actor = Actor(obs_dim=self.obs_dim, act_dim=self.act_dim,
                          hidden=self.hidden_dim).to(self.device)
            actor.load_state_dict(state[cid])
            actor.eval()
            for p in actor.parameters():
                p.requires_grad = False
            actors[cid] = actor
        return actors

    def get_name(self, idx: int) -> str:
        """ckpt 文件名（无扩展名），用于日志展示。"""
        return os.path.splitext(os.path.basename(self.ckpt_paths[idx]))[0]

    @classmethod
    def from_dir(cls, dir_path: str, cooperative_ids: List[int],
                 pattern: str = "*.pt", **kwargs) -> 'RedTeamPool':
        """递归扫目录加载所有 .pt 文件。文件名排序作为索引顺序（稳定可复现）。"""
        files = sorted(glob.glob(os.path.join(dir_path, pattern)))
        if not files:
            raise FileNotFoundError(f"目录 {dir_path} 下没有匹配 {pattern} 的 ckpt")
        return cls(files, cooperative_ids=cooperative_ids, **kwargs)


class BlueTeamPool:
    """管理 PSRO 产出的 BLUE 团队 ckpt。

    每个 BLUE 团队 ckpt 格式：
      {
        "adversarial_actors": {bid: state_dict},
        "critic": state_dict,                     # CTDE 共享 critic (AgentConditionedCritic)
        "log_std": {bid: float},
        "adversarial_ids": [...],
        "trained_against_sigma_red": [...],       # 训练时的 σ_red 快照
        "psro_iter": int,
        "args": dict,                              # 训练超参
      }
    """

    def __init__(self, save_dir: str, adversarial_ids: List[int],
                 obs_dim: int = 19, act_dim: int = 3, hidden_dim: int = 128,
                 device: str = 'cpu'):
        self.save_dir = os.path.abspath(save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        self.adversarial_ids = list(adversarial_ids)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)
        self.ckpt_paths: List[str] = []

    def __len__(self) -> int:
        return len(self.ckpt_paths)

    def next_save_path(self, psro_iter: int) -> str:
        return os.path.join(self.save_dir, f"psro{psro_iter:02d}_blue_team.pt")

    def add(self, ckpt_dict: Dict, psro_iter: int) -> str:
        """保存 BLUE 团队 ckpt 并加入池。返回保存路径。"""
        path = self.next_save_path(psro_iter)
        torch.save(ckpt_dict, path)
        self.ckpt_paths.append(path)
        return path

    def get_actors(self, idx: int) -> Dict[int, Actor]:
        """返回第 idx 个 BLUE 团队的 {bid: Actor (eval, frozen)}。"""
        if idx < 0 or idx >= len(self):
            raise IndexError(f"BlueTeamPool 索引越界: {idx} (size={len(self)})")
        ckpt = torch.load(self.ckpt_paths[idx], map_location=self.device)
        adv_state = ckpt["adversarial_actors"]
        actors: Dict[int, Actor] = {}
        for bid in self.adversarial_ids:
            if bid not in adv_state:
                raise KeyError(f"BLUE ckpt {self.ckpt_paths[idx]} 缺少 agent {bid}")
            actor = Actor(obs_dim=self.obs_dim, act_dim=self.act_dim,
                          hidden=self.hidden_dim).to(self.device)
            actor.load_state_dict(adv_state[bid])
            actor.eval()
            for p in actor.parameters():
                p.requires_grad = False
            actors[bid] = actor
        return actors

    def get_critic(self, idx: int) -> Optional[AgentConditionedCritic]:
        """返回第 idx 个 BLUE 团队的共享 critic（eval, frozen）。"""
        ckpt = torch.load(self.ckpt_paths[idx], map_location=self.device)
        critic_state = ckpt.get("critic")
        if critic_state is None:
            return None
        # state_dim 从 critic 第一个 weight 形状反推
        first_w = critic_state.get("net.0.weight")
        if first_w is None:
            return None
        in_dim = first_w.shape[1]
        num_agents = len(self.adversarial_ids)
        state_dim = in_dim - num_agents
        critic = AgentConditionedCritic(state_dim=state_dim,
                                        num_agents=num_agents,
                                        hidden=self.hidden_dim).to(self.device)
        critic.load_state_dict(critic_state)
        critic.eval()
        for p in critic.parameters():
            p.requires_grad = False
        return critic

    def get_name(self, idx: int) -> str:
        return os.path.splitext(os.path.basename(self.ckpt_paths[idx]))[0]

    def get_meta(self, idx: int) -> Dict:
        """返回第 idx 个 BLUE 团队的训练元信息（不含权重张量），用于泛化诊断。

        关键字段：
          - trained_against_sigma_red: 该 BLUE 训练时面对的 σ_red 快照（与 RED 池索引对齐）
          - psro_iter / stats / args
        """
        ckpt = torch.load(self.ckpt_paths[idx], map_location='cpu')
        return {
            "trained_against_sigma_red": ckpt.get("trained_against_sigma_red"),
            "psro_iter": ckpt.get("psro_iter"),
            "stats": ckpt.get("stats", {}),
            "args": ckpt.get("args", {}),
        }

    @classmethod
    def from_dir(cls, dir_path: str, adversarial_ids: List[int],
                 pattern: str = "psro*_blue_team.pt", **kwargs) -> 'BlueTeamPool':
        """加载已存在的 BLUE 池（如复现 JPC 评估时使用）。"""
        files = sorted(glob.glob(os.path.join(dir_path, pattern)))
        if not files:
            raise FileNotFoundError(f"目录 {dir_path} 下没有匹配 {pattern} 的 BLUE 团队 ckpt")
        pool = cls(save_dir=dir_path, adversarial_ids=adversarial_ids, **kwargs)
        pool.ckpt_paths = files
        return pool
