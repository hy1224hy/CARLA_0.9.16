"""
PSRO 元策略求解器
==================
在经验收益矩阵上求 σ_red, σ_blue。两档实现：

- UniformSolver : 均匀分布（fictitious play 风格） — baseline, 30 行
- PRDSolver    : Projected Replicator Dynamics —— 论文 §3.3 + 附录 A.3

约定（重要）：
  payoff_matrix 形状 = (n_red, n_blue)
  payoff_matrix[i, j] = "BLUE_j 对抗 RED_i" 的收益（数值越大对 BLUE 越好）
  典型口径：adv_effectiveness = 1 - coop_success_rate

  RED 收益取负：U_red = -payoff_matrix（近似零和处理；一般和场景下作为近似可用）。

  论文实际是一般和但其求解器把"对手收益分布"视作元变量，PRD 在 BLUE/RED
  各自单纯形上做投影复制动态。我们沿用相同近似。
"""
import numpy as np
from typing import Tuple


class UniformSolver:
    """均匀分布元策略 — 等价于 fictitious play 中"对手全历史等权混合"。

    最简单的基线：每个 RED 同概率被采样，每个 BLUE 同概率上场。
    不依赖 payoff 矩阵的数值，仅依赖数量。
    """

    name = "uniform"

    def solve(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_red, n_blue = payoff_matrix.shape
        if n_red == 0 or n_blue == 0:
            return np.ones(max(n_red, 1)) / max(n_red, 1), np.ones(max(n_blue, 1)) / max(n_blue, 1)
        return np.ones(n_red) / n_red, np.ones(n_blue) / n_blue


class PRDSolver:
    """Projected Replicator Dynamics（论文 §3.3）。

    复制动态:  dπ_i/dt = π_i · (f_i - <f>)
        其中 f_i 是策略 i 在当前对手分布下的期望适应度

    投影:    每步把 π 投影到 γ-探索单纯形（要求每个策略概率 ≥ γ/(K+1)）
        γ 越大 → BLUE 越鲁棒（保留弱策略权重）
        γ 越小 → BLUE 越偏向剥削最优响应

    论文实验推荐:
      - γ=0.1 → 最强剥削
      - γ=0.4 → 最稳鲁棒
      - 起步用 γ=0.2 折中

    收敛终止: 两步 σ 的 L1 差 < tol，或达到 max_iter。
    """

    name = "prd"

    def __init__(self, gamma: float = 0.2, lr: float = 0.01,
                 max_iter: int = 10000, tol: float = 1e-5):
        if not (0.0 <= gamma < 1.0):
            raise ValueError(f"gamma 必须在 [0, 1) 内，得到 {gamma}")
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    @staticmethod
    def _project(p: np.ndarray, lower: float) -> np.ndarray:
        """投影到单纯形 + 下界约束：先 clip 到 lower，再归一化。

        注：这是一种简化投影，不是 L2 最近投影，但在 PRD 语境下足够（论文也用类似实现）。
        """
        p = np.clip(p, lower, None)
        s = p.sum()
        if s <= 0:
            n = p.shape[0]
            return np.ones(n) / n
        return p / s

    def solve(self, payoff_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_red, n_blue = payoff_matrix.shape
        if n_red == 0 or n_blue == 0:
            return (np.ones(max(n_red, 1)) / max(n_red, 1),
                    np.ones(max(n_blue, 1)) / max(n_blue, 1))

        # 处理 NaN：未评估的位置用矩阵均值填充作为先验
        if np.isnan(payoff_matrix).any():
            mean_val = float(np.nanmean(payoff_matrix)) if not np.isnan(payoff_matrix).all() else 0.0
            U = np.nan_to_num(payoff_matrix, nan=mean_val)
        else:
            U = payoff_matrix.astype(np.float64)

        sigma_red = np.ones(n_red) / n_red
        sigma_blue = np.ones(n_blue) / n_blue
        lower_red = self.gamma / (n_red + 1.0)
        lower_blue = self.gamma / (n_blue + 1.0)

        for _ in range(self.max_iter):
            # BLUE 适应度：U^T @ sigma_red，每个 BLUE 策略在当前 RED 分布下的期望收益
            f_blue = U.T @ sigma_red                    # (n_blue,)
            # RED 适应度（零和近似）：-U @ sigma_blue
            f_red = -U @ sigma_blue                     # (n_red,)

            avg_f_blue = float(sigma_blue @ f_blue)
            avg_f_red = float(sigma_red @ f_red)

            # 复制动态更新
            d_blue = sigma_blue * (f_blue - avg_f_blue)
            d_red = sigma_red * (f_red - avg_f_red)

            new_blue = self._project(sigma_blue + self.lr * d_blue, lower_blue)
            new_red = self._project(sigma_red + self.lr * d_red, lower_red)

            diff = float(np.abs(new_blue - sigma_blue).sum() + np.abs(new_red - sigma_red).sum())
            sigma_blue, sigma_red = new_blue, new_red
            if diff < self.tol:
                break

        return sigma_red, sigma_blue


def make_meta_solver(name: str, **kwargs):
    """工厂方法，便于命令行参数选择。"""
    name = name.lower().strip()
    if name == "uniform":
        return UniformSolver()
    if name == "prd":
        return PRDSolver(**kwargs)
    raise ValueError(f"未知的 meta_solver 名称: {name} (支持 uniform | prd)")
