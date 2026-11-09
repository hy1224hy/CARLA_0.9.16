"""
PSRO 外层主循环
================
程序入口。装配 RED 池 + BLUE 池 + meta solver + payoff table，
按论文 Algorithm 1 的循环跑：
    每轮:
       ① 用当前 σ_red 训一个新 BLUE oracle (调 train_blue_oracle)
       ② 评估新 BLUE 对所有 RED，填充 payoff
       ③ 调 meta solver 更新 σ_red, σ_blue
       ④ JPC R_- + TB 日志

设计文档：docs/对抗训练_MAPPO_PSRO_设计计划.md
"""
import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

import argparse
import datetime as _dt
import os
import random as _rand
from typing import Dict, List

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from environment.carla_multi_agent_env import CarlaMultiAgentEnv

from training.train_multi_adversarial_mappo_v2 import (
    _scenario_agents, _load_config_file, make_adversarial_positions,
)

from training.psro.population import RedTeamPool, BlueTeamPool
from training.psro.meta_solver import make_meta_solver
from training.psro.payoff_table import PayoffTable, compute_jpc_R_minus
from training.psro.train_blue_oracle import train_blue_oracle
from training.psro.jpc_eval import evaluate_pair, prepare_eval_scenario


def set_seed(seed: int):
    _rand.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"PSRO 主循环启动 (device={device})")

    cooperative_ids = [int(x) for x in args.cooperative_ids.split(',')]
    adversarial_ids = [int(x) for x in args.adversarial_ids.split(',')]
    total_agents = len(cooperative_ids) + len(adversarial_ids)

    # ── 输出目录 + TB ──
    run_tag = _dt.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_seed{args.seed}"
    run_dir = os.path.join(os.path.abspath(args.save_dir), run_tag)
    blue_pool_dir = os.path.join(run_dir, 'blue_pool')
    tb_dir = os.path.join(run_dir, 'tb_logs')
    os.makedirs(blue_pool_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    tb_writer = SummaryWriter(tb_dir)
    print(f"输出目录: {run_dir}")

    # ── RED 池 ──
    red_pool = RedTeamPool.from_dir(
        args.red_pool_dir, cooperative_ids=cooperative_ids,
        hidden_dim=args.hidden_dim, device=str(device),
    )
    if len(red_pool) < 2:
        print(f"⚠️ RED 池只有 {len(red_pool)} 个 ckpt，PSRO 退化为 InRL（无种群效应）。"
              f"建议先按 Phase 0 扩 RED 池到 ≥ 3 个。")

    # ── BLUE 池 + 收益张量 ──
    blue_pool = BlueTeamPool(
        save_dir=blue_pool_dir, adversarial_ids=adversarial_ids,
        hidden_dim=args.hidden_dim, device=str(device),
    )
    payoff = PayoffTable(n_red=len(red_pool), n_blue=args.psro_iters)

    # ── 元策略求解器 ──
    if args.meta_solver == "prd":
        meta = make_meta_solver("prd", gamma=args.prd_gamma, lr=args.prd_lr,
                                max_iter=args.prd_max_iter, tol=args.prd_tol)
    else:
        meta = make_meta_solver(args.meta_solver)
    print(f"meta_solver = {meta.name}")

    sigma_red = np.ones(len(red_pool)) / len(red_pool)
    sigma_blue: np.ndarray = np.array([])

    # ── env（共享给所有 PSRO 轮，避免反复重启 CARLA）──
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

    # ── 准备固定场景（PSRO 默认单场景跑，避免课程干扰元策略信号）──
    scenario_dir = args.scenario_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data', 'training_scenarios'
    )
    scenario_config = prepare_eval_scenario(
        args.scenario, scenario_dir, cooperative_ids, adversarial_ids,
        adv_margin_ratio=args.adv_margin_ratio, adv_min_gap=args.adv_min_gap,
    )
    print(f"PSRO 场景: {args.scenario} (adv_min_gap={args.adv_min_gap}m)")
    print(f"  goal_zone: {scenario_config.get('goal_zone', '<未声明>')}")
    agents_final = _scenario_agents(scenario_config)
    for k, v in sorted(agents_final.items()):
        role = "BLUE" if int(k.replace("agent", "")) in adversarial_ids else "RED"
        print(f"  {k} ({role}): x={v.get('x'):.2f}, y={v.get('y'):.2f}")

    try:
        # ============================================================
        # PSRO 主循环
        # ============================================================
        for psro_iter in range(args.psro_iters):
            print(f"\n{'='*70}")
            print(f"PSRO iter {psro_iter}/{args.psro_iters-1}")
            print(f"{'='*70}")
            print(f"σ_red = [{', '.join(f'{x:.3f}' for x in sigma_red)}]")

            # ─── 步①：训练新 BLUE oracle ───
            print(f"\n>>> Step ①: 训练新 BLUE oracle，{args.inner_updates} updates")
            inner_args = _build_inner_args(args)
            blue_team_ckpt = train_blue_oracle(
                env=env,
                red_pool=red_pool,
                sigma_red=sigma_red,
                adversarial_ids=adversarial_ids,
                cooperative_ids=cooperative_ids,
                scenario_config=scenario_config,
                args=inner_args,
                psro_iter=psro_iter,
                tb_writer=tb_writer,
                tb_step_offset=psro_iter * args.inner_updates,
            )
            save_path = blue_pool.add(blue_team_ckpt, psro_iter)
            print(f"  → BLUE 团队保存: {save_path}")

            # ─── 步②：评估新 BLUE 对所有 RED ───
            print(f"\n>>> Step ②: 评估 BLUE[{psro_iter}] vs 所有 RED ({args.n_eval_episodes} ep/对)")
            new_blue_actors = blue_pool.get_actors(psro_iter)
            for ri in range(len(red_pool)):
                red_actors = red_pool.get_actors(ri)
                result = evaluate_pair(
                    env, red_actors, new_blue_actors,
                    cooperative_ids, adversarial_ids, scenario_config,
                    n_episodes=args.n_eval_episodes,
                    threat_radius=args.threat_radius,
                    threat_lateral=args.threat_lateral,
                    threat_tau=args.threat_tau,
                    threat_near=args.threat_near,
                    max_steps=args.max_steps,
                    device=str(device),
                )
                payoff.fill(ri, psro_iter, result,
                            n_episodes=args.n_eval_episodes, scenarios=[args.scenario])
                print(f"  RED[{ri}] {red_pool.get_name(ri):<20s}: "
                      f"coop_succ={result['coop_success_rate']:.0%}, "
                      f"blue_coll={result['blue_collision_rate']:.0%}, "
                      f"threat={result['blue_threat_window_pct']:.0%}, "
                      f"slowdown={result['red_relative_slowdown']:+.0%}")

            # ─── 步③：更新元策略 ───
            print(f"\n>>> Step ③: 更新元策略")
            current_payoff = payoff.submatrix(n_red=len(red_pool), n_blue=psro_iter + 1)
            sigma_red, sigma_blue = meta.solve(current_payoff)
            print(f"  new σ_red  = [{', '.join(f'{x:.3f}' for x in sigma_red)}]")
            print(f"  new σ_blue = [{', '.join(f'{x:.3f}' for x in sigma_blue)}]")

            # ─── 步④：JPC + TB ───
            # JPC 需方阵，仅当 len(red) == psro_iter+1 时计算
            jpc_R = float('nan')
            if len(red_pool) == psro_iter + 1:
                jpc_R = compute_jpc_R_minus(current_payoff)
                print(f"  JPC R_- = {jpc_R:+.3f}")

            tb_writer.add_scalar('psro/iter', psro_iter, psro_iter)
            tb_writer.add_scalar('psro/blue_pool_size', len(blue_pool), psro_iter)
            avg_adv_eff = float(np.nanmean(current_payoff))
            tb_writer.add_scalar('psro/avg_adv_effectiveness', avg_adv_eff, psro_iter)
            if not np.isnan(jpc_R):
                tb_writer.add_scalar('psro/jpc_R_minus', jpc_R, psro_iter)
            for ri in range(len(red_pool)):
                tb_writer.add_scalar(f'psro/sigma_red_{ri}', float(sigma_red[ri]), psro_iter)
            for bi in range(len(sigma_blue)):
                tb_writer.add_scalar(f'psro/sigma_blue_{bi}', float(sigma_blue[bi]), psro_iter)

            # ─── 持久化 payoff ───
            payoff.save_json(
                os.path.join(run_dir, "payoff_table.json"),
                extra={
                    "psro_iter": psro_iter,
                    "sigma_red": [float(x) for x in sigma_red],
                    "sigma_blue": [float(x) for x in sigma_blue],
                    "jpc_R_minus": jpc_R,
                    "red_labels": [red_pool.get_name(i) for i in range(len(red_pool))],
                },
            )
    finally:
        env.close()
        tb_writer.close()

    print(f"\nPSRO 完成。输出: {run_dir}")
    print(f"  BLUE 池: {blue_pool_dir} ({len(blue_pool)} teams)")
    print(f"  payoff: {os.path.join(run_dir, 'payoff_table.json')}")
    print(f"\n后续步骤：")
    print(f"  python -m training.psro.jpc_eval \\")
    print(f"      --red-pool-dir {args.red_pool_dir} \\")
    print(f"      --blue-pool-dir {blue_pool_dir} \\")
    print(f"      --scenario {args.scenario} \\")
    print(f"      --output-dir {os.path.join(run_dir, 'jpc')}")


def _build_inner_args(outer_args):
    """从 outer 参数装一个 namespace 给 train_blue_oracle。"""
    return argparse.Namespace(
        # 训练规模
        inner_updates=outer_args.inner_updates,
        rollout_episodes=outer_args.rollout_episodes,
        max_steps=outer_args.max_steps,
        min_transitions_per_update=outer_args.min_transitions_per_update,
        # PPO 超参
        ppo_epochs=outer_args.ppo_epochs,
        minibatch_size=outer_args.minibatch_size,
        lr_actor=outer_args.lr_actor,
        lr_critic=outer_args.lr_critic,
        gamma=outer_args.gamma,
        gae_lambda=outer_args.gae_lambda,
        clip_eps=outer_args.clip_eps,
        ent_coef=outer_args.ent_coef,
        max_grad_norm=outer_args.max_grad_norm,
        target_kl=outer_args.target_kl,
        lr_decay=outer_args.lr_decay,
        # 模型
        hidden_dim=outer_args.hidden_dim,
        init_log_std=outer_args.init_log_std,
        # 新方案
        beta_kl=outer_args.beta_kl,
        threat_coef=outer_args.threat_coef,
        threat_radius=outer_args.threat_radius,
        threat_lateral=outer_args.threat_lateral,
        threat_tau=outer_args.threat_tau,
        threat_near=outer_args.threat_near,
        target_blue_collision_rate=outer_args.target_blue_collision_rate,
        self_collision_coef_init=outer_args.self_collision_coef_init,
        self_collision_coef_max=outer_args.self_collision_coef_max,
    )


def build_parser():
    p = argparse.ArgumentParser(description="PSRO 外层主循环（MAPPO oracle + KL 锚定）")

    # ─── PSRO 核心 ───
    p.add_argument("--red-pool-dir", type=str, required=True,
                   help="目录下放 RED ckpt（每个含 cooperative actors）")
    p.add_argument("--psro-iters", type=int, default=5,
                   help="PSRO 外循环轮次（每轮产出一个新 BLUE 团队）")
    p.add_argument("--inner-updates", type=int, default=30,
                   help="每轮 BLUE oracle 训练 PPO update 数")
    p.add_argument("--n-eval-episodes", type=int, default=5,
                   help="每对 (RED_i, BLUE_j) 评估的 episode 数")
    p.add_argument("--meta-solver", type=str, default="prd",
                   choices=["uniform", "prd"])
    p.add_argument("--prd-gamma", type=float, default=0.2,
                   help="PRD 探索下界，0.1 偏剥削 / 0.4 偏鲁棒")
    p.add_argument("--prd-lr", type=float, default=0.01)
    p.add_argument("--prd-max-iter", type=int, default=10000)
    p.add_argument("--prd-tol", type=float, default=1e-5)

    # ─── 场景 / 团队 ───
    p.add_argument("--scenario", type=str, default="scenario_01_basic.json")
    p.add_argument("--scenario-dir", type=str, default="")
    p.add_argument("--cooperative-ids", type=str, default="0,1")
    p.add_argument("--adversarial-ids", type=str, default="2,3")
    p.add_argument("--adv-margin-ratio", type=float, default=0.15)
    p.add_argument("--adv-min-gap", type=float, default=14.0,
                   help="相邻车最小纵向间距(米)，拉开生成间距防一上来追尾；0=旧紧凑布局。"
                        "评估(jpc_eval/baseline_eval)须用同值")
    p.add_argument("--adversary-collision-ends-episode", action="store_true", default=False,
                   help="修改1：默认 False=蓝车自撞不结束回合(despawn 该车,红车继续)，消除自杀漏洞；"
                        "加此开关=恢复旧行为(任一碰撞即全局终止)")

    # ─── 内层 MAPPO 超参 ───
    p.add_argument("--rollout-episodes", type=int, default=16)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--min-transitions-per-update", type=int, default=512)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.15)
    p.add_argument("--ent-coef", type=float, default=0.03)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=0.02)
    p.add_argument("--lr-decay", action="store_true", default=True)
    p.add_argument("--no-lr-decay", dest="lr_decay", action="store_false")
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--init-log-std", type=float, default=-0.5)

    # ─── KL 锚定 + 风险增量 + 自适应自碰 ───
    p.add_argument("--beta-kl", type=float, default=0.1)
    # 优化 A：威胁奖励改"安全挡道带"后，降低 threat_coef 权重、收紧自碰目标、抬高自碰罚上限，
    # 让"无碰撞逼停"成为最优而非"怼上去撞"。
    p.add_argument("--threat-coef", type=float, default=1.0)   # was 2.0
    p.add_argument("--threat-radius", type=float, default=15.0)
    p.add_argument("--threat-lateral", type=float, default=2.5)
    p.add_argument("--threat-tau", type=float, default=1.5)    # 旧核用，安全带核已不依赖
    p.add_argument("--threat-near", type=float, default=3.0,
                   help="优化 A：安全挡道带近端(米)。纵向 < 此值=撞距离，不给威胁分")
    p.add_argument("--target-blue-collision-rate", type=float, default=0.03)  # was 0.05
    p.add_argument("--self-collision-coef-init", type=float, default=1.0)
    p.add_argument("--self-collision-coef-max", type=float, default=20.0,
                   help="优化 A：自适应自碰罚系数上限(was 10)")

    # ─── CARLA / 其他 ───
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--town", type=str, default="Town02")
    p.add_argument("--straight-road-length", type=float, default=100.0)
    p.add_argument("--timestep", type=float, default=0.1)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, default="./logs/psro_runs")

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
