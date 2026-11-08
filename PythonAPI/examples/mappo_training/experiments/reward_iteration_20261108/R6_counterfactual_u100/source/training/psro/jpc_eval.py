"""
JPC (Joint Policy Correlation) 交叉评估
=========================================
为 RED 池 × BLUE 池中所有配对运行评估，构造 |R|×|B| 收益矩阵，
输出热图 + R_- 指标（论文 §3.2 (a)）。

R_- = (diag_mean - off_diag_mean) / diag_mean
  > 0.30 严重过拟合
  0.10-0.30 有过拟合迹象
  < 0.10 泛化良好

可作为 psro_runner 完成后的独立诊断工具运行：
  python -m training.psro.jpc_eval \\
      --red-pool-dir logs/red_pool/ckpts \\
      --blue-pool-dir logs/psro_runs/<id>/blue_pool \\
      --output-dir logs/psro_runs/<id>/jpc \\
      --scenario scenario_01_basic.json \\
      --n-eval 5
"""
import sys
from pathlib import Path
_pkg_root = Path(__file__).resolve().parent.parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.append(str(_pkg_root))

import argparse
import math
import os
import json
from typing import Dict, List, Optional

import numpy as np
import torch

from environment.carla_multi_agent_env import CarlaMultiAgentEnv
from models.actor_critic import Actor

from training.train_multi_adversarial_mappo_v2 import (
    _scenario_agents, _load_config_file, make_adversarial_positions, action_tanh_to_env
)
from training.train_multi_adversarial_kl_risk import compute_blue_threat_scores
from training.train_adversarial_fixed_scenario import (
    is_any_red_under_threat, avg_red_speed
)

from training.psro.population import RedTeamPool, BlueTeamPool
from training.psro.payoff_table import PayoffTable, compute_jpc_R_minus, split_diag_offdiag


def evaluate_pair(
    env: CarlaMultiAgentEnv,
    red_actors: Dict[int, Actor],
    blue_actors: Dict[int, Actor],
    cooperative_ids: List[int],
    adversarial_ids: List[int],
    scenario_config: Dict,
    n_episodes: int,
    threat_radius: float = 15.0,
    threat_lateral: float = 2.5,
    threat_tau: float = 1.5,
    threat_near: float = 3.0,
    max_steps: int = 1000,
    device: str = 'cpu',
) -> Dict[str, float]:
    """单对 (RED, BLUE) 评估，返回多指标聚合 dict。"""
    env.set_scenario(scenario_config)
    total_agents = len(cooperative_ids) + len(adversarial_ids)

    stats = {
        "coop_success": 0, "coop_collision": 0, "blue_collision": 0,
        "n_episodes": 0,
        "threat_window_pcts": [], "red_speeds_threat": [], "red_speeds_free": [],
        "avg_threat_scores": [], "ep_lengths": [],
    }

    for _ in range(n_episodes):
        obs_dict, info = env.reset()
        collision_sticky = {i: False for i in range(total_agents)}
        # 修改1：逐 agent 终止（death-masking）。eval 必须让红车跑到底以量 coop_success，
        # 故收尾只看"红车是否全终止"；蓝车自撞被 env despawn 后不再采样、也不结束回合。
        red_done = {cid: False for cid in cooperative_ids}
        blue_done = {bid: False for bid in adversarial_ids}
        steps = 0
        ep_threat_steps = 0
        ep_red_spd_threat: List[float] = []
        ep_red_spd_free: List[float] = []
        ep_threat_scores: List[float] = []

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
                    obs_t = torch.as_tensor(obs_dict[bid], dtype=torch.float32,
                                            device=device).unsqueeze(0)
                    a = blue_actors[bid].get_deterministic_action(obs_t).squeeze(0).cpu().numpy()
                    action_dict[bid] = action_tanh_to_env(a)

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

            threat_active = is_any_red_under_threat(
                env, cooperative_ids, adversarial_ids, threat_radius, threat_lateral,
            )
            if threat_active:
                ep_threat_steps += 1
            rs = avg_red_speed(env, cooperative_ids)
            if rs is not None:
                (ep_red_spd_threat if threat_active else ep_red_spd_free).append(rs)
            threat_scores = compute_blue_threat_scores(
                env, cooperative_ids, adversarial_ids,
                threat_radius=threat_radius,
                threat_lateral=threat_lateral,
                threat_tau=threat_tau,
                threat_near=threat_near,
            )
            ep_threat_scores.append(sum(threat_scores.values()))

            obs_dict = next_obs
            if all(red_done.values()):   # 红车全终止（到点/碰撞/超时）即回合结束
                break

        coop_success = all(env.reached_target.get(cid, False) for cid in cooperative_ids)
        coop_collision = any(collision_sticky[cid] for cid in cooperative_ids)
        blue_collision = any(collision_sticky[bid] for bid in adversarial_ids)
        stats["n_episodes"] += 1
        stats["coop_success"] += 1 if coop_success else 0
        stats["coop_collision"] += 1 if coop_collision else 0
        stats["blue_collision"] += 1 if blue_collision else 0
        stats["ep_lengths"].append(steps)
        stats["threat_window_pcts"].append(ep_threat_steps / max(steps, 1))
        stats["avg_threat_scores"].append(
            float(np.mean(ep_threat_scores)) if ep_threat_scores else 0.0
        )
        if ep_red_spd_threat:
            stats["red_speeds_threat"].append(float(np.mean(ep_red_spd_threat)))
        if ep_red_spd_free:
            stats["red_speeds_free"].append(float(np.mean(ep_red_spd_free)))

    n = max(stats["n_episodes"], 1)
    rs_t = float(np.mean(stats["red_speeds_threat"])) if stats["red_speeds_threat"] else float('nan')
    rs_f = float(np.mean(stats["red_speeds_free"])) if stats["red_speeds_free"] else float('nan')
    slowdown = 0.0
    if not math.isnan(rs_f) and rs_f > 0.1 and not math.isnan(rs_t):
        slowdown = (rs_f - rs_t) / rs_f

    return {
        "coop_success_rate": stats["coop_success"] / n,
        "coop_collision_rate": stats["coop_collision"] / n,
        "blue_collision_rate": stats["blue_collision"] / n,
        "blue_threat_window_pct": float(np.mean(stats["threat_window_pcts"])),
        "red_relative_slowdown": slowdown,
        "avg_threat_score": float(np.mean(stats["avg_threat_scores"]))
            if stats["avg_threat_scores"] else 0.0,
        "avg_episode_length": float(np.mean(stats["ep_lengths"])),
    }


def prepare_eval_scenario(scenario_name: str, scenario_dir: str,
                          cooperative_ids: List[int], adversarial_ids: List[int],
                          adv_margin_ratio: float = 0.15, adv_min_gap: float = 0.0) -> Dict:
    config = _load_config_file(scenario_name, scenario_dir)
    if config is None:
        raise FileNotFoundError(f"场景 {scenario_name} 加载失败")
    config = dict(config)
    config["agents"] = make_adversarial_positions(
        config, cooperative_ids, adversarial_ids,
        margin_ratio=adv_margin_ratio, min_gap=adv_min_gap,
    )
    return config


def compute_generalization_report(adv_eff: np.ndarray,
                                  red_labels: List[str],
                                  blue_labels: List[str],
                                  blue_pool=None) -> Dict:
    """泛化 / 鲁棒性诊断（不依赖"对角线配对"假设，因此对 PSRO 池始终有意义）。

    背景：论文的 JPC R_- 要求矩阵对角线是"配对协同训练"的实例（InRL 基线满足：
    blue_i 专门对 red_i 训）。但 PSRO 的 BLUE 是对 σ_red **混合分布**训出来的，
    blue_j 与 red_j 并无配对关系，故 PSRO 池上的 R_- 不能按"联合策略相关性"解读。
    这里给出一组与索引配对无关、对 BLUE 适用的鲁棒性/泛化指标：

      per_blue.mean_eff   : 该 BLUE 对整个 RED 池的平均对抗有效性（越高越通用）
      per_blue.worst_eff  : 该 BLUE 对最难对手的有效性 = min_i M[i,j]（最坏情况鲁棒性）
      per_blue.std_eff    : 跨 RED 的方差（越小越稳定，过拟合单一 RED 时会偏大）
      per_blue.gen_gap    : 训练分布有效性 − 全池平均有效性
                            = Σ_i σ_red_train[i]·M[i,j] − mean_i M[i,j]
                            > 0 越大 → 越是"只在自己训练的 σ_red 上强"，泛化越差

      best_single_blue_worstcase : max_j min_i M[i,j]
                                   "单个最佳 BLUE 的最坏情况有效性"——衡量是否存在
                                   一个对所有 RED 都不差的通用对手
      uniform_mix_worstcase      : min_i mean_j M[i,j]
                                   "BLUE 池均匀混合"对最难 RED 的有效性（混合是否抬高下界）
    """
    nr, nb = adv_eff.shape
    M = np.nan_to_num(adv_eff, nan=np.nanmean(adv_eff)) if np.isnan(adv_eff).any() else adv_eff

    per_blue = []
    for j in range(nb):
        col = M[:, j]
        entry = {
            "blue": blue_labels[j] if j < len(blue_labels) else f"blue{j}",
            "mean_eff": float(col.mean()),
            "worst_eff": float(col.min()),
            "best_eff": float(col.max()),
            "std_eff": float(col.std()),
            "gen_gap": None,
        }
        # 泛化间隙：需要该 BLUE 训练时的 σ_red 快照（长度须与 RED 池一致）
        if blue_pool is not None:
            try:
                meta = blue_pool.get_meta(j)
                sr = meta.get("trained_against_sigma_red")
                if sr is not None and len(sr) == nr:
                    sr = np.asarray(sr, dtype=np.float64)
                    if sr.sum() > 0:
                        sr = sr / sr.sum()
                        train_eff = float(sr @ col)
                        entry["train_eff"] = train_eff
                        entry["gen_gap"] = float(train_eff - col.mean())
            except Exception:
                pass
        per_blue.append(entry)

    best_single_worst = float(np.max([np.min(M[:, j]) for j in range(nb)])) if nb else float('nan')
    uniform_mix_per_red = M.mean(axis=1) if nb else np.array([])
    uniform_mix_worst = float(uniform_mix_per_red.min()) if uniform_mix_per_red.size else float('nan')

    return {
        "per_blue": per_blue,
        "best_single_blue_worstcase": best_single_worst,
        "uniform_mix_worstcase": uniform_mix_worst,
        "note": ("worst_eff/gen_gap 是 PSRO 池的核心泛化指标；R_- 仅在 BLUE 池与 RED 池"
                 "按索引配对（InRL 基线）时才可按论文 JPC 解读。"),
    }


def save_heatmap(matrix: np.ndarray, path: str, title: str = "",
                 row_labels: Optional[List[str]] = None,
                 col_labels: Optional[List[str]] = None,
                 value_fmt: str = ".2f"):
    """保存矩阵热图 PNG。matplotlib 缺失时跳过。"""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  ⚠️ matplotlib 未安装，跳过热图 {path}")
        return False
    nr, nc = matrix.shape
    fig, ax = plt.subplots(figsize=(max(6, nc * 1.0), max(5, nr * 0.8)))
    safe = np.nan_to_num(matrix, nan=0.0)
    im = ax.imshow(safe, cmap='RdYlGn_r', aspect='auto', vmin=0.0, vmax=1.0)
    if row_labels:
        ax.set_yticks(range(nr))
        ax.set_yticklabels(row_labels, fontsize=8)
    if col_labels:
        ax.set_xticks(range(nc))
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel("BLUE team")
    ax.set_ylabel("RED team")
    if title:
        ax.set_title(title)
    for i in range(nr):
        for j in range(nc):
            v = matrix[i, j]
            txt = "NA" if (isinstance(v, float) and np.isnan(v)) else format(v, value_fmt)
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='white' if safe[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    cooperative_ids = [int(x) for x in args.cooperative_ids.split(',')]
    adversarial_ids = [int(x) for x in args.adversarial_ids.split(',')]
    print(f"团队: cooperative={cooperative_ids}, adversarial={adversarial_ids}")

    red_pool = RedTeamPool.from_dir(
        args.red_pool_dir, cooperative_ids=cooperative_ids,
        hidden_dim=args.hidden_dim, device=str(device),
    )
    blue_pool = BlueTeamPool.from_dir(
        args.blue_pool_dir, adversarial_ids=adversarial_ids,
        hidden_dim=args.hidden_dim, device=str(device),
    )
    print(f"RED 池: {len(red_pool)} | BLUE 池: {len(blue_pool)}")

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
        num_agents=len(cooperative_ids) + len(adversarial_ids),
        straight_road_length=args.straight_road_length,
        timestep=args.timestep, max_steps=args.max_steps,
        spawn_vehicles=True,
        adversarial_agent_ids=adversarial_ids,
        reward_version="v2",
        disable_tensorboard=True,
        seed=args.seed,
        adversary_collision_ends_episode=args.adversary_collision_ends_episode,
    )

    payoff = PayoffTable(len(red_pool), len(blue_pool))
    try:
        for ri in range(len(red_pool)):
            red_actors = red_pool.get_actors(ri)
            for bi in range(len(blue_pool)):
                blue_actors = blue_pool.get_actors(bi)
                print(f"\n>>> RED[{ri}] {red_pool.get_name(ri)}  vs  "
                      f"BLUE[{bi}] {blue_pool.get_name(bi)}")
                result = evaluate_pair(
                    env, red_actors, blue_actors,
                    cooperative_ids, adversarial_ids, scenario_config,
                    n_episodes=args.n_eval,
                    threat_radius=args.threat_radius,
                    threat_lateral=args.threat_lateral,
                    threat_tau=args.threat_tau,
                    threat_near=args.threat_near,
                    max_steps=args.max_steps,
                    device=str(device),
                )
                payoff.fill(ri, bi, result, n_episodes=args.n_eval, scenarios=[args.scenario])
                print(f"    coop_succ={result['coop_success_rate']:.0%}, "
                      f"blue_coll={result['blue_collision_rate']:.0%}, "
                      f"threat={result['blue_threat_window_pct']:.0%}, "
                      f"slowdown={result['red_relative_slowdown']:.0%}")
    finally:
        env.close()

    # ── 输出 ──
    os.makedirs(args.output_dir, exist_ok=True)
    adv_eff = payoff.adv_effectiveness_matrix()
    payoff.save_json(os.path.join(args.output_dir, "payoff_table.json"))

    row_labels = [red_pool.get_name(i) for i in range(len(red_pool))]
    col_labels = [blue_pool.get_name(i) for i in range(len(blue_pool))]

    # adv_effectiveness 热图（BLUE 视角，越红越好）
    save_heatmap(adv_eff,
                 os.path.join(args.output_dir, "adv_effectiveness_heatmap.png"),
                 title="Adversary Effectiveness = 1 - coop_success",
                 row_labels=row_labels, col_labels=col_labels)
    # coop_success 热图（baseline 视角，越绿越好）
    save_heatmap(payoff.metrics["coop_success_rate"],
                 os.path.join(args.output_dir, "coop_success_heatmap.png"),
                 title="Cooperative Success Rate",
                 row_labels=row_labels, col_labels=col_labels)
    # blue_collision 热图（自毁度量）
    save_heatmap(payoff.metrics["blue_collision_rate"],
                 os.path.join(args.output_dir, "blue_collision_heatmap.png"),
                 title="Blue (Adversary) Collision Rate",
                 row_labels=row_labels, col_labels=col_labels)

    R_minus = compute_jpc_R_minus(adv_eff)
    diag_vals, offdiag_vals = (None, None)
    if adv_eff.shape[0] == adv_eff.shape[1] and adv_eff.shape[0] >= 2:
        diag_vals, offdiag_vals = split_diag_offdiag(adv_eff)

    # 对 PSRO BLUE 池始终有意义的泛化/鲁棒性诊断（不依赖对角线配对）
    gen = compute_generalization_report(adv_eff, row_labels, col_labels, blue_pool=blue_pool)
    # R_- 是否可按论文 JPC 解读：仅当方阵 + BLUE 与 RED 索引配对（InRL 基线）。
    # PSRO BLUE 训的是 σ_red 混合，blue_j 与 red_j 无配对关系 → 标注为"仅供参考"。
    r_minus_valid_as_jpc = bool(args.paired_pool) and not math.isnan(R_minus)

    report = {
        "jpc_R_minus": R_minus,
        "jpc_R_minus_valid_as_jpc": r_minus_valid_as_jpc,
        "diag_mean": float(np.mean(diag_vals)) if diag_vals is not None else None,
        "off_diag_mean": float(np.mean(offdiag_vals)) if offdiag_vals is not None else None,
        "interpretation": (
            (("严重过拟合" if R_minus > 0.30 else
              "有过拟合迹象" if R_minus > 0.10 else
              "泛化良好") if r_minus_valid_as_jpc else
             "R_- 仅在配对池(--paired-pool, InRL 基线)下按 JPC 解读；PSRO 池请看 generalization_report")
        ) if not math.isnan(R_minus) else "N/A (非方阵)",
        "generalization": gen,
        "red_labels": row_labels,
        "blue_labels": col_labels,
        "adv_effectiveness_matrix": [
            [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in row]
            for row in adv_eff
        ],
        "n_eval_episodes_per_pair": args.n_eval,
        "scenario": args.scenario,
    }
    with open(os.path.join(args.output_dir, "jpc_report.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 64)
    print(f"JPC 评估完成 → {args.output_dir}")
    if r_minus_valid_as_jpc:
        print(f"  [配对池] R_- = {R_minus:.3f}  ({report['interpretation']})")
        if diag_vals is not None:
            print(f"    配对(对角线) 平均 adv_eff = {float(np.mean(diag_vals)):.3f}")
            print(f"    错配(非对角) 平均 adv_eff = {float(np.mean(offdiag_vals)):.3f}")
    else:
        print(f"  R_- = {R_minus:.3f}  (非配对池，不按 JPC 解读；见下方泛化指标)")
    print(f"  ── 泛化 / 鲁棒性（PSRO 核心）──")
    print(f"    单最优 BLUE 最坏情况有效性 best_single_worstcase = {gen['best_single_blue_worstcase']:.3f}")
    print(f"    均匀混合最坏情况有效性     uniform_mix_worstcase  = {gen['uniform_mix_worstcase']:.3f}")
    for e in gen["per_blue"]:
        gap = f"{e['gen_gap']:+.3f}" if e["gen_gap"] is not None else "  n/a"
        print(f"    {e['blue']:<22s} mean={e['mean_eff']:.3f} worst={e['worst_eff']:.3f} "
              f"std={e['std_eff']:.3f} gen_gap={gap}")
    print("=" * 64)


def build_parser():
    p = argparse.ArgumentParser(description="JPC 交叉评估（RED 池 × BLUE 池）")
    p.add_argument("--red-pool-dir", type=str, required=True)
    p.add_argument("--blue-pool-dir", type=str, required=True)
    p.add_argument("--scenario", type=str, default="scenario_01_basic.json")
    p.add_argument("--scenario-dir", type=str, default="")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--n-eval", type=int, default=5)
    p.add_argument("--cooperative-ids", type=str, default="0,1")
    p.add_argument("--adversarial-ids", type=str, default="2,3")
    p.add_argument("--paired-pool", action="store_true",
                   help="BLUE 池与 RED 池按索引配对(InRL 基线: blue_i 专门对 red_i 训)，"
                        "此时 R_- 才可按论文 JPC 解读；PSRO 池请勿加此开关")

    p.add_argument("--threat-radius", type=float, default=15.0)
    p.add_argument("--threat-lateral", type=float, default=2.5)
    p.add_argument("--threat-tau", type=float, default=1.5)
    p.add_argument("--threat-near", type=float, default=3.0,
                   help="优化 A：安全挡道带近端(米)，须与训练一致")
    p.add_argument("--adv-margin-ratio", type=float, default=0.15)
    p.add_argument("--adv-min-gap", type=float, default=14.0,
                   help="相邻车最小纵向间距(米)，拉开生成间距防一上来追尾；0=旧紧凑布局。"
                        "须与训练(psro_runner --adv-min-gap)一致")
    p.add_argument("--adversary-collision-ends-episode", action="store_true", default=False,
                   help="修改1：默认 False=蓝车自撞不结束回合；须与训练一致")

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
