"""
基线评估：判定 coop_success≡0 的归因（红车基础策略 vs 蓝车自杀式局部最优）
==================================================================================
背景见 docs/对抗训练_PSRO_首次运行问题诊断_20260622.md。

PSRO 首跑里所有 (RED, BLUE) 配对 coop_success 都是 0、蓝车 100% 撞车、回合 ~30 步即终止。
两种可能根因需要区分：
  (A) 红车在 4 车场景里本就过不去（红车基础策略不足 / 生成过挤 / 终点判定）；
  (B) 红车本能过，是被「对抗蓝车自杀式撞车」瞬间终止回合（蓝车收敛到局部最优）。

本工具把蓝车换成**良性驾驶**（不再对抗），在**同一 4 车场景**评估每个 RED：
  --blue-mode natural （默认）：蓝车用 RED 的合作驾驶策略开向终点 = 4 车自然车流 → 判 A/B 的关键
  --blue-mode stationary       ：蓝车刹停（单车道必堵，纯几何对照，不能判 A/B）
  --blue-mode passive          ：蓝车恒定小油门直行（无目标）

判读：
  natural 下 coop_success 高（≥0.6）         → (B) 几何/红车 OK，问题在对抗蓝车自杀
  natural 下 coop_success 低 且 coop_collision 高 → (A) 红车自然车流就撞（不会跟车）/生成过挤
  natural 下 coop_success 低 且 coop_collision 低 → (A') 终点判定/距离/被堵

用法（mappo_training/ 下，pipenv，先启 CARLA）：
  python -m training.psro.baseline_eval \
      --red-pool-dir logs/red_pool/ckpts \
      --scenario scenario_01_basic.json \
      --blue-mode natural --n-eval 5 \
      --output-dir logs/psro_runs/baseline
"""
import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor

from training.train_multi_adversarial_mappo_v2 import action_tanh_to_env
from training.psro.population import RedTeamPool
from training.psro.jpc_eval import prepare_eval_scenario


# 良性蓝车的非自然驾驶动作（env 格式 [throttle, steer, brake]）
_STATIONARY_ACTION = np.array([0.0, 0.0, 1.0], dtype=np.float32)   # 全刹车
_PASSIVE_ACTION = np.array([0.4, 0.0, 0.0], dtype=np.float32)      # 小油门直行


def evaluate_baseline(
    env: CarlaMultiAgentEnv,
    red_actors: Dict[int, Actor],
    driver_actor: Actor,
    cooperative_ids: List[int],
    adversarial_ids: List[int],
    scenario_config: Dict,
    n_episodes: int,
    blue_mode: str,
    max_steps: int = 1000,
    device: str = 'cpu',
) -> Dict[str, float]:
    """单个 RED 在良性蓝车下的评估。driver_actor 仅 blue_mode='natural' 时使用。"""
    env.set_scenario(scenario_config)
    total_agents = len(cooperative_ids) + len(adversarial_ids)

    n_coop_succ = 0
    n_coop_coll = 0
    n_blue_coll = 0
    n_blue_succ = 0
    ep_lengths: List[int] = []

    for _ in range(n_episodes):
        obs_dict, info = env.reset()
        collision_sticky = {i: False for i in range(total_agents)}
        # 修改1：逐 agent 终止（death-masking）；收尾看红车是否全终止（蓝车自撞被 despawn，不结束回合）
        red_done = {cid: False for cid in cooperative_ids}
        blue_done = {bid: False for bid in adversarial_ids}
        steps = 0
        for _ in range(max_steps):
            steps += 1
            action_dict: Dict[int, np.ndarray] = {}
            with torch.no_grad():
                for cid in cooperative_ids:
                    if red_done[cid] or cid not in obs_dict:
                        continue
                    obs_t = torch.as_tensor(obs_dict[cid], dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    a = red_actors[cid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                    action_dict[cid] = action_tanh_to_env(a)
                for bid in adversarial_ids:
                    if blue_done[bid] or bid not in obs_dict:
                        continue
                    if blue_mode == 'natural':
                        obs_t = torch.as_tensor(obs_dict[bid], dtype=torch.float32,
                                                device=device).unsqueeze(0)
                        a = driver_actor.get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                        action_dict[bid] = action_tanh_to_env(a)
                    elif blue_mode == 'stationary':
                        action_dict[bid] = _STATIONARY_ACTION.copy()
                    else:  # passive
                        action_dict[bid] = _PASSIVE_ACTION.copy()

            next_obs, _, dones, info = env.step(action_dict)
            step_collisions = info.get('collisions', {}) or {}
            for i in range(total_agents):
                if step_collisions.get(i, False):
                    collision_sticky[i] = True
            for cid in cooperative_ids:
                if dones.get(cid, False):
                    red_done[cid] = True
            for bid in adversarial_ids:
                if dones.get(bid, False):
                    blue_done[bid] = True
            obs_dict = next_obs
            if all(red_done.values()):
                break

        coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
        coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
        blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)
        blue_success = all(env.reached_target.get(bid, False) for bid in adversarial_ids)
        n_coop_succ += 1 if coop_success else 0
        n_coop_coll += 1 if coop_collision else 0
        n_blue_coll += 1 if blue_collision else 0
        n_blue_succ += 1 if blue_success else 0
        ep_lengths.append(steps)

    n = max(n_episodes, 1)
    return {
        "coop_success_rate": n_coop_succ / n,
        "coop_collision_rate": n_coop_coll / n,
        "blue_collision_rate": n_blue_coll / n,
        "blue_success_rate": n_blue_succ / n,
        "avg_episode_length": float(np.mean(ep_lengths)) if ep_lengths else 0.0,
    }


def _verdict(mean_coop_succ: float, mean_coop_coll: float, blue_mode: str) -> str:
    if blue_mode != 'natural':
        return (f"[{blue_mode}] 仅作对照，不能判 A/B（单车道下 stationary 必堵）。"
                "请用 --blue-mode natural 做归因判定。")
    if mean_coop_succ >= 0.6:
        return ("(B) 几何/红车基础策略 OK —— 良性蓝车下红车能到终点。"
                "PSRO 里 coop_success≡0 是【对抗蓝车收敛到自杀式撞车的局部最优】。"
                "处置：抬 --threat-coef、提高 --self-collision-coef-init、或改终止/几何让"
                "「无碰撞挡道」成为可行最优。")
    if mean_coop_coll >= 0.5:
        return ("(A) 红车基础策略不足 —— 即便蓝车良性自然行驶，红车也大量碰撞"
                "（不会跟车/避让）或生成过挤。处置：先修红车策略 / 拉开生成间距"
                "（--adv-margin-ratio 调大或换 scenario_03_far）。")
    return ("(A') 红车在自然车流下不撞但仍到不了终点 —— 可能终点判定/距离过远/被前车堵。"
            "处置：检查 goal_zone、车间距、max_steps。")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    cooperative_ids = [int(x) for x in args.cooperative_ids.split(',')]
    adversarial_ids = [int(x) for x in args.adversarial_ids.split(',')]
    total_agents = len(cooperative_ids) + len(adversarial_ids)
    print(f"基线评估: blue_mode={args.blue_mode}, cooperative={cooperative_ids}, "
          f"adversarial={adversarial_ids}")

    red_pool = RedTeamPool.from_dir(
        args.red_pool_dir, cooperative_ids=cooperative_ids,
        hidden_dim=args.hidden_dim, device=str(device),
    )

    scenario_dir = args.scenario_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data', 'training_scenarios'
    )
    scenario_config = prepare_eval_scenario(
        args.scenario, scenario_dir, cooperative_ids, adversarial_ids,
        adv_margin_ratio=args.adv_margin_ratio, adv_min_gap=args.adv_min_gap,
    )

    env = CarlaMultiAgentEnv(
        host=args.host, port=args.port, town=args.town,
        num_agents=total_agents,
        straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps,
        spawn_vehicles=True,
        adversarial_agent_ids=adversarial_ids,
        reward_version="v2",
        disable_tensorboard=True,
        seed=args.seed,
        adversary_collision_ends_episode=args.adversary_collision_ends_episode,
    )

    per_red = []
    try:
        for ri in range(len(red_pool)):
            red_actors = red_pool.get_actors(ri)
            # natural 模式：蓝车用同一 RED 的合作驾驶策略当良性司机
            driver_actor = red_actors[cooperative_ids[0]]
            result = evaluate_baseline(
                env, red_actors, driver_actor,
                cooperative_ids, adversarial_ids, scenario_config,
                n_episodes=args.n_eval, blue_mode=args.blue_mode,
                max_steps=args.max_steps, device=str(device),
            )
            result["red"] = red_pool.get_name(ri)
            per_red.append(result)
            print(f"  RED[{ri}] {red_pool.get_name(ri):<28s}: "
                  f"coop_succ={result['coop_success_rate']:.0%}, "
                  f"coop_coll={result['coop_collision_rate']:.0%}, "
                  f"blue_coll={result['blue_collision_rate']:.0%}, "
                  f"blue_succ={result['blue_success_rate']:.0%}, "
                  f"ep_len={result['avg_episode_length']:.0f}")
    finally:
        env.close()

    mean_succ = float(np.mean([r["coop_success_rate"] for r in per_red])) if per_red else 0.0
    mean_coll = float(np.mean([r["coop_collision_rate"] for r in per_red])) if per_red else 0.0
    verdict = _verdict(mean_succ, mean_coll, args.blue_mode)

    os.makedirs(args.output_dir, exist_ok=True)
    report = {
        "blue_mode": args.blue_mode,
        "scenario": args.scenario,
        "n_eval_per_red": args.n_eval,
        "cooperative_ids": cooperative_ids,
        "adversarial_ids": adversarial_ids,
        "per_red": per_red,
        "mean_coop_success_rate": mean_succ,
        "mean_coop_collision_rate": mean_coll,
        "verdict": verdict,
    }
    out_path = os.path.join(args.output_dir, f"baseline_{args.blue_mode}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"基线评估完成 → {out_path}")
    print(f"  均值: coop_success={mean_succ:.0%}, coop_collision={mean_coll:.0%}")
    print(f"  判定: {verdict}")
    print("=" * 70)


def build_parser():
    p = argparse.ArgumentParser(description="基线评估：良性蓝车下判定红车基础策略 vs 蓝车自杀")
    p.add_argument("--red-pool-dir", type=str, required=True)
    p.add_argument("--scenario", type=str, default="scenario_01_basic.json")
    p.add_argument("--scenario-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, default="logs/psro_runs/baseline")
    p.add_argument("--blue-mode", type=str, default="natural",
                   choices=["natural", "stationary", "passive"])
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--cooperative-ids", type=str, default="0,1")
    p.add_argument("--adversarial-ids", type=str, default="2,3")
    p.add_argument("--adv-margin-ratio", type=float, default=0.15)
    p.add_argument("--adv-min-gap", type=float, default=14.0,
                   help="相邻车最小纵向间距(米)，拉开生成间距防一上来追尾；0=旧紧凑布局")
    p.add_argument("--adversary-collision-ends-episode", action="store_true", default=False,
                   help="修改1：默认 False=蓝车自撞不结束回合(与 PSRO 训练一致)")

    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town02")
    p.add_argument("--straight-road-length", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
