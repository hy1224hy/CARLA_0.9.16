"""
多对抗 MAPPO 评估脚本
=====================
兼容 train_multi_adversarial_mappo_v2.py 保存的 ckpt（adversarial_actors 复数 dict，
任意 M 个对抗 agent），在多个场景上做 **有/无对抗** 对照评估，输出统一的
对抗有效性指标，供实验计划表 6.3 / 6.4 填写。

核心指标:
    adversary_effectiveness = coop_success(无对抗) − coop_success(有对抗)

两种模式（车流密度保持一致，只切换对抗 agent 的策略）:
    baseline : 对抗 agent 用合作策略（正常前向驾驶）→ coop_success_baseline
    attacked : 对抗 agent 用训练好的对抗策略           → coop_success_attacked

用法:
    python -m evaluation.evaluate_multi_adversarial \
        --pretrained-ckpt <coop.pt> \
        --adversarial-ckpt <multi_adv_v2 best.pt> \
        --num-episodes-per-scenario 5 \
        --output-file ./logs/exp/eval/B0_attacked.json
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import os
import json
import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def action_tanh_to_env(action_tanh: np.ndarray) -> np.ndarray:
    throttle_raw = (action_tanh[0] + 1.0) * 0.5
    steer = action_tanh[1]
    brake_raw = (action_tanh[2] + 1.0) * 0.5
    if throttle_raw >= brake_raw:
        throttle, brake = throttle_raw, 0.0
    else:
        throttle, brake = 0.0, brake_raw
    return np.array([throttle, steer, brake], dtype=np.float32)


@dataclass
class ModeStats:
    """单一模式（baseline / attacked）在某场景上的聚合统计"""
    n_episodes: int = 0
    coop_success: int = 0          # 所有合作 agent 都到达
    coop_collision: int = 0        # 任一合作 agent 碰撞
    completion_times: List[float] = field(default_factory=list)  # 合作团队全部到达时的用时(s)

    @property
    def coop_success_rate(self) -> float:
        return self.coop_success / max(self.n_episodes, 1)

    @property
    def coop_collision_rate(self) -> float:
        return self.coop_collision / max(self.n_episodes, 1)

    @property
    def avg_completion_time(self) -> float:
        return float(np.mean(self.completion_times)) if self.completion_times else float('nan')


class MultiAdversarialEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

        # 加载多对抗 ckpt，读取团队划分
        adv_path = self._normalize_path(args.adversarial_ckpt)
        if not os.path.exists(adv_path):
            raise FileNotFoundError(f"对抗 checkpoint 不存在: {adv_path}")
        adv_ckpt = torch.load(adv_path, map_location=self.device)
        self.cooperative_ids: List[int] = adv_ckpt.get("cooperative_ids")
        self.adversarial_ids: List[int] = adv_ckpt.get("adversarial_ids")
        if self.cooperative_ids is None or self.adversarial_ids is None:
            raise KeyError("对抗 ckpt 缺少 cooperative_ids / adversarial_ids（请用 v2 训练脚本生成）")
        self.total_agents = len(self.cooperative_ids) + len(self.adversarial_ids)
        print(f"团队划分: 合作 {self.cooperative_ids} | 对抗 {self.adversarial_ids}")

        # 对抗 actors（复数 dict）
        adv_actors_state = adv_ckpt.get("adversarial_actors")
        if not adv_actors_state:
            raise KeyError("对抗 ckpt 缺少 adversarial_actors（复数）")
        self.adversarial_actors: Dict[int, Actor] = {}
        for aid in self.adversarial_ids:
            if aid not in adv_actors_state:
                raise KeyError(f"对抗 ckpt 缺少 agent {aid} 权重")
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(self.device)
            actor.load_state_dict(adv_actors_state[aid])
            actor.eval()
            self.adversarial_actors[aid] = actor

        # 合作 actors（从预训练 coop ckpt）
        coop_path = self._normalize_path(args.pretrained_ckpt)
        if not os.path.exists(coop_path):
            raise FileNotFoundError(f"合作 checkpoint 不存在: {coop_path}")
        coop_ckpt = torch.load(coop_path, map_location=self.device)
        coop_state = coop_ckpt.get("actors", {})
        self.cooperative_actors: Dict[int, Actor] = {}
        for cid in self.cooperative_ids:
            if cid not in coop_state:
                raise KeyError(f"合作 ckpt 缺少 agent {cid} 权重")
            actor = Actor(obs_dim=19, act_dim=3, hidden=args.hidden_dim).to(self.device)
            actor.load_state_dict(coop_state[cid])
            actor.eval()
            self.cooperative_actors[cid] = actor
        # baseline 模式下对抗 agent 借用的"正常驾驶"策略
        self.normal_policy = self.cooperative_actors[self.cooperative_ids[0]]
        print(f"已加载合作 actors {list(self.cooperative_actors.keys())}、对抗 actors {list(self.adversarial_actors.keys())}")

    @staticmethod
    def _normalize_path(p: str) -> str:
        while p.startswith('\\') or p.startswith('/'):
            p = p[1:]
        return p.replace('\\', os.sep).replace('/', os.sep)

    def _make_env(self) -> CarlaMultiAgentEnv:
        return CarlaMultiAgentEnv(
            host=self.args.host, port=self.args.port, town=self.args.town,
            num_agents=self.total_agents,
            straight_road_length=self.args.straight_road_length,
            timestep=self.args.timestep, max_steps=self.args.max_steps,
            spawn_vehicles=True,
            adversarial_agent_ids=self.adversarial_ids,
            reward_version="v2",
            disable_tensorboard=True,
        )

    def _run_episode(self, env: CarlaMultiAgentEnv, has_adversary: bool) -> Dict:
        obs_dict, info = env.reset()
        collision_sticky = {i: False for i in range(self.total_agents)}
        reach_step = {i: None for i in range(self.total_agents)}
        steps = 0
        for step in range(self.args.max_steps):
            steps += 1
            action_dict: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for cid in self.cooperative_ids:
                    if cid not in obs_dict:
                        continue
                    obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32, device=self.device).unsqueeze(0)
                    a = self.cooperative_actors[cid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                    action_dict[cid] = action_tanh_to_env(a)
                for aid in self.adversarial_ids:
                    if aid not in obs_dict:
                        continue
                    policy = self.adversarial_actors[aid] if has_adversary else self.normal_policy
                    obs_t = torch.as_tensor(obs_dict[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                    a = policy.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                    action_dict[aid] = action_tanh_to_env(a)

            next_obs, rewards, dones, info = env.step(action_dict)
            step_collisions = info.get('collisions', {}) or {}
            for i in range(self.total_agents):
                if step_collisions.get(i, False):
                    collision_sticky[i] = True
                if reach_step[i] is None and env.reached_target.get(i, False):
                    reach_step[i] = steps
            obs_dict = next_obs
            if all(dones.values()):
                break

        coop_success = all(env.reached_target.get(cid, False) for cid in self.cooperative_ids)
        coop_collision = any(collision_sticky[cid] for cid in self.cooperative_ids)
        # 合作团队全部到达时的用时 = 最后一个到达 agent 的步数 × timestep
        completion_time = None
        if coop_success:
            last = max(reach_step[cid] for cid in self.cooperative_ids)
            completion_time = last * self.args.timestep
        return {
            "coop_success": coop_success,
            "coop_collision": coop_collision,
            "completion_time": completion_time,
            "steps": steps,
        }

    def _eval_scenario(self, env, scenario_name, scenario_dir) -> Dict[str, ModeStats]:
        config = env.load_scenario_by_name(scenario_name, scenario_dir)
        if config is None:
            config = env.load_random_scenario(scenario_dir, [scenario_name])
        if config:
            env.set_scenario(config)

        stats = {"baseline": ModeStats(), "attacked": ModeStats()}
        for mode, has_adv in [("baseline", False), ("attacked", True)]:
            st = stats[mode]
            for _ in range(self.args.num_episodes_per_scenario):
                r = self._run_episode(env, has_adversary=has_adv)
                st.n_episodes += 1
                st.coop_success += 1 if r["coop_success"] else 0
                st.coop_collision += 1 if r["coop_collision"] else 0
                if r["completion_time"] is not None:
                    st.completion_times.append(r["completion_time"])
        return stats

    def evaluate(self) -> Dict:
        scenario_dir = self.args.test_scenario_dir or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'training_scenarios')
        scenario_files = sorted([f for f in os.listdir(scenario_dir) if f.endswith('.json')])
        if self.args.scenarios:
            wanted = set(self.args.scenarios.split(','))
            scenario_files = [f for f in scenario_files if f in wanted]
        print(f"评估场景目录: {scenario_dir}，共 {len(scenario_files)} 个场景")

        env = self._make_env()
        per_scenario: Dict[str, Dict] = {}
        agg = {"baseline": ModeStats(), "attacked": ModeStats()}
        try:
            for sname in scenario_files:
                print(f"\n>>> 场景 {sname}")
                stats = self._eval_scenario(env, sname, scenario_dir)
                eff = stats["baseline"].coop_success_rate - stats["attacked"].coop_success_rate
                per_scenario[sname] = {
                    "baseline_coop_success": stats["baseline"].coop_success_rate,
                    "attacked_coop_success": stats["attacked"].coop_success_rate,
                    "effectiveness": eff,
                    "baseline_coop_collision": stats["baseline"].coop_collision_rate,
                    "attacked_coop_collision": stats["attacked"].coop_collision_rate,
                    "baseline_completion_time": stats["baseline"].avg_completion_time,
                    "attacked_completion_time": stats["attacked"].avg_completion_time,
                }
                for mode in ("baseline", "attacked"):
                    agg[mode].n_episodes += stats[mode].n_episodes
                    agg[mode].coop_success += stats[mode].coop_success
                    agg[mode].coop_collision += stats[mode].coop_collision
                    agg[mode].completion_times.extend(stats[mode].completion_times)
                print(f"    baseline coop_success={stats['baseline'].coop_success_rate:.2%}, "
                      f"attacked={stats['attacked'].coop_success_rate:.2%}, "
                      f"effectiveness={eff:+.2%}")
        finally:
            env.close()

        overall_eff = agg["baseline"].coop_success_rate - agg["attacked"].coop_success_rate
        result = {
            "pretrained_ckpt": self.args.pretrained_ckpt,
            "adversarial_ckpt": self.args.adversarial_ckpt,
            "cooperative_ids": self.cooperative_ids,
            "adversarial_ids": self.adversarial_ids,
            "num_episodes_per_scenario": self.args.num_episodes_per_scenario,
            "overall": {
                "baseline_coop_success": agg["baseline"].coop_success_rate,
                "attacked_coop_success": agg["attacked"].coop_success_rate,
                "effectiveness": overall_eff,
                "baseline_coop_collision": agg["baseline"].coop_collision_rate,
                "attacked_coop_collision": agg["attacked"].coop_collision_rate,
                "baseline_completion_time": agg["baseline"].avg_completion_time,
                "attacked_completion_time": agg["attacked"].avg_completion_time,
            },
            "per_scenario": per_scenario,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        self._print_summary(result)
        return result

    def _print_summary(self, result: Dict):
        o = result["overall"]
        print("\n" + "=" * 64)
        print("多对抗评估汇总")
        print("=" * 64)
        print(f"  无对抗  合作成功率: {o['baseline_coop_success']:.2%}   碰撞率: {o['baseline_coop_collision']:.2%}")
        print(f"  有对抗  合作成功率: {o['attacked_coop_success']:.2%}   碰撞率: {o['attacked_coop_collision']:.2%}")
        print(f"  ★ 对抗有效性 (effectiveness): {o['effectiveness']:+.2%}")
        print("=" * 64)


def build_parser():
    parser = argparse.ArgumentParser(description="多对抗 MAPPO 评估（有/无对抗对照）")
    parser.add_argument("--pretrained-ckpt", type=str, required=True, help="合作团队预训练 ckpt")
    parser.add_argument("--adversarial-ckpt", type=str, required=True, help="多对抗 v2 ckpt（含 adversarial_actors）")
    parser.add_argument("--num-episodes-per-scenario", type=int, default=5)
    parser.add_argument("--test-scenario-dir", type=str, default="", help="评估场景目录（默认 data/training_scenarios）")
    parser.add_argument("--scenarios", type=str, default="", help="逗号分隔的场景文件名子集，留空=全部")
    parser.add_argument("--output-file", type=str, default="", help="结果 JSON 输出路径")

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    set_seed(args.seed)
    evaluator = MultiAdversarialEvaluator(args)
    result = evaluator.evaluate()
    if args.output_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"结果已保存: {args.output_file}")
