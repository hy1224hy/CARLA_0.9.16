"""
PSRO (Policy-Space Response Oracles) 对抗训练框架
==================================================
基于 Lanctot et al., NeurIPS 2017 的 PSRO 算法 + 现有 MAPPO 内核（KL 锚定 + 风险增量奖励）
实现的对抗策略种群训练。

设计文档：docs/对抗训练_MAPPO_PSRO_设计计划.md

模块结构：
  population.py        - RedTeamPool / BlueTeamPool 策略池管理
  meta_solver.py       - UniformSolver / PRDSolver 元策略求解器
  payoff_table.py      - PayoffTable 经验收益张量
  train_blue_oracle.py - train_blue_oracle() 内层 MAPPO oracle 训练
  jpc_eval.py          - JPC 交叉评估矩阵（独立诊断工具）
  psro_runner.py       - PSRO 外层主循环（程序入口）

典型用法（命令行）：
  python -m training.psro.psro_runner \\
      --red-pool-dir logs/red_pool \\
      --scenario scenario_01_basic.json \\
      --psro-iters 5 --inner-updates 30

  python -m training.psro.jpc_eval \\
      --red-pool-dir logs/red_pool \\
      --blue-pool-dir logs/psro_runs/<id>/blue_pool \\
      --output-dir logs/psro_runs/<id>/jpc
"""

from .population import RedTeamPool, BlueTeamPool
from .meta_solver import UniformSolver, PRDSolver, make_meta_solver
from .payoff_table import PayoffTable, compute_jpc_R_minus

__all__ = [
    "RedTeamPool", "BlueTeamPool",
    "UniformSolver", "PRDSolver", "make_meta_solver",
    "PayoffTable", "compute_jpc_R_minus",
]
