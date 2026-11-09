"""Render the Mut-MAPPO framework diagram (figure for the mid-term report).

One-off, CARLA-independent. Output: high-resolution PNG (3200x1800) for PPT.
Re-run after editing to regenerate:

    pipenv run python utils/render_mut_mappo_framework_diagram.py

Layout is a vertical closed loop drawn in a 100 x 56.5 unit canvas
(y increases downward): CARLA env -> observation/reward interface ->
team learners A/B -> action protocol -> back to env, with the mutual
distillation gate and deployment stage below the learners.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "MutMAPPO框架图_20261108.png"

# ---------------------------------------------------------------- palette
C_ENV = dict(fc="#F0F5F1", ec="#5B7A5E")
C_RED = dict(fc="#F9E8E8", ec="#B85C5C")
C_SCN = dict(fc="#F7F2E5", ec="#A08A55")
C_BLU = dict(fc="#E8EEF7", ec="#4A6FA5")
C_STRIP = dict(fc="#F4F4F4", ec="#8A8A8A")
C_DIST = dict(fc="#F1EDF8", ec="#7B68A6")
C_DEP = dict(fc="#F4F4F4", ec="#666666")
ARROW = "#4A5568"


def box(ax, x, y, w, h, title, lines, fc, ec, title_fs=10.5, fs=8.5,
        lw=1.6, title_pad=1.5, rounding=0.6):
    """Rounded box; (x, y) is the TOP-left corner, y grows downward."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
        linewidth=lw, facecolor=fc, edgecolor=ec, clip_on=False))
    if title:
        ax.text(x + w / 2, y + title_pad, title, ha="center", va="top",
                fontsize=title_fs, fontweight="bold", color="#222222")
    if lines:
        body_top = y + title_pad + (title_fs / 6.0 if title else 0)
        body = "\n".join(lines)
        ax.text(x + w / 2, body_top, body, ha="center", va="top",
                fontsize=fs, color="#333333", linespacing=1.9)


def arrow(ax, p1, p2, ls="-", lw=1.5, color=ARROW, rad=0.0, both=False,
          shrink=0.0):
    style = "<|-|>" if both else "-|>"
    ax.annotate("", xy=p2, xytext=p1, xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                linestyle=ls, shrinkA=shrink, shrinkB=shrink,
                                connectionstyle=f"arc3,rad={rad}"),
                annotation_clip=False)


def label(ax, xy, text, fs=8, color="#4A5568", dx=0, dy=0, ha="center",
          rotation=0):
    ax.text(xy[0] + dx, xy[1] + dy, text, ha=ha, va="center", fontsize=fs,
            color=color, rotation=rotation)


def main(out: Path = OUT):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=200)
    ax.set_xlim(0, 100)
    ax.set_ylim(56.5, 0)          # inverted: y grows downward
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ------------------------------------------------ row 1: CARLA env
    box(ax, 5, 0.5, 90, 4.6,
        "Mut-MAPPO：CARLA 0.9.16 四车对抗互蒸馏框架（同步模式 · dt=0.1s · 上限 500 步）",
        None, C_ENV["fc"], C_ENV["ec"], title_fs=12, title_pad=1.35)
    box(ax, 7, 1.6, 24, 2.9, "红方（冻结 · 只推理）",
        ["红0 / 红1 · 基础驾驶策略", "R8C1-v6 checkpoint"],
        C_RED["fc"], C_RED["ec"], title_fs=10, fs=8.5, title_pad=0.55)
    box(ax, 33, 1.6, 34, 2.9, "场景族 × 双向（seed 扰动）",
        ["blocking / cut_in / parallel / staggered",
         "红与蓝物理交互（碰撞 / TTC 几何）"],
        C_SCN["fc"], C_SCN["ec"], title_fs=10, fs=8.5, title_pad=0.55)
    box(ax, 69, 1.6, 24, 2.9, "蓝方（学习 · A/B 轮流控制）",
        ["蓝2 / 蓝3 · 以红策略权重初始化", "每回合由当前学习器采样"],
        C_BLU["fc"], C_BLU["ec"], title_fs=10, fs=8.5, title_pad=0.55)

    # ------------------------------------------------ row 2: data interface
    box(ax, 5, 6.3, 90, 3.0, "数据接口",
        ["局部观测 o_i ∈ R20（8 基础 + 11 射线 + 剩余时间）× 4 → 全局状态 s ∈ R80",
         "个体奖励 r_i（R3 画像）= 0.3·v6软项 + v6硬项 + 违规成本 + 个体对抗几何"],
        C_STRIP["fc"], C_STRIP["ec"], title_fs=10, fs=8.5, title_pad=0.5)
    arrow(ax, (50, 5.1), (50, 6.3))
    label(ax, (52.5, 5.7), "观测 / 奖励")

    # ------------------------------------------------ row 3: learners A/B
    team_lines = [
        "Actor×2：20 → 128 → 128 → (μ, logσ)，tanh 高斯，logσ∈[−2,0]",
        "团队 Critic（blue_team）：蓝方槽位 40＋局部 ID 2 → 128 → 128 → 1 ＋ ValueNorm（可迁移初始化）",
        "PPO：GAE(γ=.999,λ=.95) → 优势标准化 → ×2 epochs → clip .15＋熵 .01 → KL 早停(>.03)",
        "Critic：Huber δ=10 · 梯度裁剪 0.5 · Adam 3e-4 线性衰减",
    ]
    box(ax, 5, 11.3, 42, 17.0, "学习器 A（角色 2/3 · 独立参数 / 优化器 / 随机流）",
        team_lines, C_BLU["fc"], C_BLU["ec"], title_fs=10.5, fs=8.5, title_pad=1.1)
    box(ax, 53, 11.3, 42, 17.0, "学习器 B（角色 2/3 · 独立参数 / 优化器 / 随机流）",
        team_lines, C_BLU["fc"], C_BLU["ec"], title_fs=10.5, fs=8.5, title_pad=1.1)
    arrow(ax, (25, 9.3), (26, 11.3))
    arrow(ax, (75, 9.3), (74, 11.3))
    label(ax, (24.5, 10.3), "o, s, r", dx=0, dy=0)
    label(ax, (74.5, 10.3), "o, s, r")
    # A <-> B teacher exchange (dashed, double-headed)
    arrow(ax, (47, 19.8), (53, 19.8), ls="--", both=True)
    label(ax, (50, 18.7), "教师快照\n＋门控", fs=8)

    # ------------------------------------------------ row 4: action protocol
    box(ax, 5, 30.3, 90, 3.5, "动作协议（signed-longitudinal 死区 v2）",
        ["a = [l, steer] ∈ [−1,1]² → g(l)=sign(l)·max(|l|−0.05,0)/0.95"
         " → 油门/刹车互斥 → [throttle, steer, brake]"],
        C_STRIP["fc"], C_STRIP["ec"], title_fs=10, fs=8.5, title_pad=0.5)
    arrow(ax, (27, 28.3), (30, 30.3))
    arrow(ax, (73, 28.3), (70, 30.3))
    label(ax, (28.5, 29.2), "动作 a")
    label(ax, (71.5, 29.2), "动作 a")
    # return path back into the env (right margin)
    arrow(ax, (96.8, 32.05), (96.8, 5.1))
    label(ax, (96.8, 18), "控制命令", rotation=90, dx=-0.3)

    # ------------------------------------------------ row 5: distillation
    box(ax, 18, 35.3, 64, 11.0,
        "互蒸馏（selective 模式 · β=0.01 · 方向 student→teacher）",
        ["每 warmup 5 / refresh 10 轮：gate 集确定性评估 A、B → 配对 n / q / b 统计",
         "场景族门控 g_f = 1[n≥8 ∧ b_t≤0.125 ∧ b_t≤b_s ∧ q_t−q_s≥0.25] → 冻结教师快照（stopgrad）",
         "辅助损失 L = L_PPO + β·mean( g·KL(π学生 ‖ π教师) )",
         "off = 无互蒸馏（MAPPO 基线） / plain = 全量蒸馏 / selective = 门控蒸馏"],
        C_DIST["fc"], C_DIST["ec"], title_fs=10.5, fs=8.5, title_pad=1.1)
    arrow(ax, (41, 19.8), (35, 35.3), ls="--")
    arrow(ax, (59, 19.8), (65, 35.3), ls="--")
    label(ax, (34, 33.5), "证据统计", fs=8)
    label(ax, (66, 33.5), "证据统计", fs=8)

    # ------------------------------------------------ row 6: deployment
    box(ax, 36, 48.3, 28, 3.5, "部署（训练后）",
        ["门控择优冻结一支团队", "→ C0/C1/C2 确定性评估"],
        C_DEP["fc"], C_DEP["ec"], title_fs=10, fs=8.5, title_pad=0.55)
    arrow(ax, (50, 46.3), (50, 48.3))
    label(ax, (53, 47.3), "选择")

    fig.savefig(out, dpi=200, facecolor="white", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    main(Path(args.out))
