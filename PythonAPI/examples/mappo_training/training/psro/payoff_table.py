"""
经验收益张量
==============
存所有 (RED_i, BLUE_j) 配对的多维评估指标。

主指标（用于 meta_solver）：
  adv_effectiveness = 1 - coop_success_rate    ← BLUE 视角收益，数值越高对 BLUE 越好

副指标（用于诊断）：
  coop_collision_rate
  blue_collision_rate
  blue_threat_window_pct
  red_relative_slowdown
  avg_threat_score
  avg_episode_length

JPC R_- 指标（论文 §3.2 (a)）：
  R_- = (diag_mean - off_diag_mean) / diag_mean
  - 矩阵形状必须方阵 |R| == |B|
  - 数值越高表示过拟合越严重
"""
import os
import json
import datetime
from typing import Dict, List, Optional

import numpy as np


class PayoffTable:
    """收益张量。多维指标各占一张矩阵，未评估的位置存 NaN。

    形状 = (n_red, n_blue)，rows = RED，cols = BLUE。
    """

    METRIC_KEYS = [
        "coop_success_rate",
        "coop_collision_rate",
        "blue_collision_rate",
        "blue_threat_window_pct",
        "red_relative_slowdown",
        "avg_threat_score",
        "avg_episode_length",
    ]

    def __init__(self, n_red: int, n_blue: int):
        self.n_red = int(n_red)
        self.n_blue = int(n_blue)
        self.metrics: Dict[str, np.ndarray] = {
            k: np.full((self.n_red, self.n_blue), np.nan, dtype=np.float64)
            for k in self.METRIC_KEYS
        }
        # 评估元信息
        self.eval_n_episodes = np.zeros((self.n_red, self.n_blue), dtype=np.int32)
        self.eval_scenarios: List[List[str]] = [[None] * self.n_blue for _ in range(self.n_red)]

    def fill(self, red_idx: int, blue_idx: int, results: Dict,
             n_episodes: int = 0, scenarios: Optional[List[str]] = None):
        """填入一对 (RED_i, BLUE_j) 的评估结果。

        results 应包含 METRIC_KEYS 子集；缺失键保留 NaN。
        """
        if not (0 <= red_idx < self.n_red and 0 <= blue_idx < self.n_blue):
            raise IndexError(f"索引 ({red_idx},{blue_idx}) 越界，形状=({self.n_red},{self.n_blue})")
        for k in self.METRIC_KEYS:
            if k in results and results[k] is not None and not (isinstance(results[k], float) and np.isnan(results[k])):
                self.metrics[k][red_idx, blue_idx] = float(results[k])
        if n_episodes > 0:
            self.eval_n_episodes[red_idx, blue_idx] = int(n_episodes)
        if scenarios:
            self.eval_scenarios[red_idx][blue_idx] = list(scenarios)

    def adv_effectiveness_matrix(self) -> np.ndarray:
        """BLUE 视角对抗收益矩阵 = 1 - coop_success_rate。

        未评估位置保持 NaN，PRDSolver 内部会处理。
        """
        return 1.0 - self.metrics["coop_success_rate"]

    def submatrix(self, n_red: Optional[int] = None, n_blue: Optional[int] = None) -> np.ndarray:
        """取左上角子矩阵（用于 PSRO 中尚未填满 max_blue 时）。"""
        nr = n_red if n_red is not None else self.n_red
        nb = n_blue if n_blue is not None else self.n_blue
        return self.adv_effectiveness_matrix()[:nr, :nb]

    def to_dict(self) -> Dict:
        """序列化为 dict，NaN 转 None 便于 JSON 兼容。"""
        out = {
            "n_red": self.n_red,
            "n_blue": self.n_blue,
            "metrics": {},
            "eval_n_episodes": self.eval_n_episodes.tolist(),
        }
        for k in self.METRIC_KEYS:
            arr = self.metrics[k]
            out["metrics"][k] = [
                [None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in row]
                for row in arr
            ]
        return out

    def save_json(self, path: str, extra: Optional[Dict] = None):
        d = self.to_dict()
        if extra:
            d.update(extra)
        d["timestamp"] = datetime.datetime.now().isoformat()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        return path

    def filled_pairs(self) -> List[tuple]:
        """返回所有已评估的 (red_idx, blue_idx) 对。"""
        m = self.metrics["coop_success_rate"]
        idx = np.argwhere(~np.isnan(m))
        return [(int(r), int(b)) for r, b in idx]

    def __repr__(self):
        succ = self.metrics["coop_success_rate"]
        filled = int(np.sum(~np.isnan(succ)))
        return (f"PayoffTable(n_red={self.n_red}, n_blue={self.n_blue}, "
                f"filled={filled}/{self.n_red*self.n_blue})")


def compute_jpc_R_minus(matrix: np.ndarray) -> float:
    """JPC R_- 指标（论文 §3.2 (a)）。

    R_- = (diag_mean - off_diag_mean) / diag_mean

    - matrix 应为 BLUE 视角收益（数值越高对 BLUE 越好）
    - 必须方阵；非方阵返回 NaN
    - diag_mean ≤ 0 时退化处理（返回 0）

    数值解读：
      R_- > 0.30: 严重过拟合
      0.10–0.30: 有过拟合迹象
      < 0.10  : 泛化良好

    论文实验中 InRL 的 R_- 在大地图上能到 71.7%。
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return float('nan')
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    # 处理 NaN
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix, nan=np.nanmean(matrix) if not np.isnan(matrix).all() else 0.0)

    diag = np.diag(matrix)
    mask = ~np.eye(n, dtype=bool)
    off_diag = matrix[mask]
    diag_mean = float(diag.mean())
    off_diag_mean = float(off_diag.mean())
    if abs(diag_mean) < 1e-9:
        return 0.0
    return (diag_mean - off_diag_mean) / diag_mean


def split_diag_offdiag(matrix: np.ndarray):
    """返回 (diag_values, off_diag_values)；用于绘图与统计。非方阵抛错。"""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"split_diag_offdiag 仅支持方阵，得到 shape={matrix.shape}")
    n = matrix.shape[0]
    diag = np.diag(matrix)
    off_diag = matrix[~np.eye(n, dtype=bool)]
    return diag, off_diag
