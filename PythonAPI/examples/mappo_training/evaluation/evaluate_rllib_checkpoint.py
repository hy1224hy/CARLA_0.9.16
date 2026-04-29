import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import os
import argparse
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from environment.carla_rllib_wrapper import RLlibCarlaEnv


def _build_eval_env_creator(render: bool):
    """注册给 RLlib 的环境创建器：强制评估时关闭额外 logger，并按需渲染。"""

    def _creator(config):
        merged = dict(config)
        merged["enable_logging"] = False
        merged["render"] = render
        return RLlibCarlaEnv(merged)

    return _creator


def run_evaluation(
    checkpoint_path: str,
    eval_rounds: int = 5,
    host: str | None = None,
    port: int | None = None,
    town: str | None = None,
    num_agents: int | None = None,
    render: bool = True,
):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")

    # 使用 RLlib 内部 runner 环境，避免脚本额外再创建一个环境实例。
    register_env("carla_env", _build_eval_env_creator(render=render))

    ray.init(ignore_reinit_error=True)

    print(f"加载 checkpoint(Algorithm): {checkpoint_path}")
    algo = Algorithm.from_checkpoint(checkpoint_path)

    # 尝试在运行时覆盖关键 env 参数（若 checkpoint 配置允许会生效）。
    if host is not None:
        algo.config.env_config["host"] = host
    if port is not None:
        algo.config.env_config["port"] = port
    if town is not None:
        algo.config.env_config["town"] = town
    if num_agents is not None:
        algo.config.env_config["num_agents"] = num_agents
    algo.config.env_config["enable_logging"] = False
    algo.config.env_config["render"] = render

    print("开始评估（使用 RLlib 内部 EnvRunner）...")
    print("--------------------------------------------------")

    try:
        for i in range(1, eval_rounds + 1):
            result = algo.evaluate()

            env_runners = result.get("env_runners", {})
            episode_return_mean = env_runners.get("episode_return_mean", None)
            episode_len_mean = env_runners.get("episode_len_mean", None)

            print(
                f"[Eval {i}/{eval_rounds}] "
                f"episode_return_mean={episode_return_mean}, "
                f"episode_len_mean={episode_len_mean}"
            )

        print("--------------------------------------------------")
        print("评估完成")

    finally:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RLlib checkpoint with internal EnvRunner (no extra env in script)"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="checkpoint 路径")
    parser.add_argument("--eval-rounds", type=int, default=5, help="调用 evaluate() 的次数")
    parser.add_argument("--town", type=str, default=None, help="可选：覆盖 checkpoint 中 town")
    parser.add_argument("--host", type=str, default=None, help="可选：覆盖 checkpoint 中 host")
    parser.add_argument("--port", type=int, default=None, help="可选：覆盖 checkpoint 中 port")
    parser.add_argument("--num-agents", type=int, default=None, help="可选：覆盖 checkpoint 中智能体数量")
    parser.add_argument("--no-render", action="store_true", help="禁用可视化")

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        eval_rounds=args.eval_rounds,
        town=args.town,
        host=args.host,
        port=args.port,
        num_agents=args.num_agents,
        render=not args.no_render,
    )
