"""
同时运行对抗训练和可视化
"""

import argparse
import subprocess
import sys
import os
import time


def main():
    parser = argparse.ArgumentParser(description="运行对抗训练并启动可视化")

    # 训练参数（传递给训练脚本）
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", type=str, default="Town02")
    parser.add_argument("--straight-road-length", type=float, default=100.0)
    parser.add_argument("--timestep", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--pretrained-ckpt", type=str, required=True)

    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--minibatch-size", type=int, default=256)

    parser.add_argument("--lr-actor", type=float, default=3e-4)
    parser.add_argument("--lr-critic", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--save-dir", type=str, default="./logs/adversarial_ppo_ckpts")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=5)

    # 可视化参数
    parser.add_argument("--viz-refresh-rate", type=float, default=0.05)
    parser.add_argument("--no-viz", action="store_true", help="不启动可视化")

    args = parser.parse_args()

    # 获取脚本目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, "train_adversarial_ppo.py")
    viz_script = os.path.join(script_dir, "visualize_adversarial_training.py")

    # 构建训练命令
    train_args = [
        sys.executable, train_script,
        "--host", args.host,
        "--port", str(args.port),
        "--town", args.town,
        "--straight-road-length", str(args.straight_road_length),
        "--timestep", str(args.timestep),
        "--max-steps", str(args.max_steps),
        "--pretrained-ckpt", args.pretrained_ckpt,
        "--updates", str(args.updates),
        "--rollout-steps", str(args.rollout_steps),
        "--ppo-epochs", str(args.ppo_epochs),
        "--minibatch-size", str(args.minibatch_size),
        "--lr-actor", str(args.lr_actor),
        "--lr-critic", str(args.lr_critic),
        "--gamma", str(args.gamma),
        "--gae-lambda", str(args.gae_lambda),
        "--clip-eps", str(args.clip_eps),
        "--ent-coef", str(args.ent_coef),
        "--max-grad-norm", str(args.max_grad_norm),
        "--hidden-dim", str(args.hidden_dim),
        "--seed", str(args.seed),
        "--save-dir", args.save_dir,
        "--save-interval", str(args.save_interval),
        "--log-interval", str(args.log_interval),
    ]

    if args.cpu:
        train_args.append("--cpu")

    # 构建可视化命令
    viz_args = [
        sys.executable, viz_script,
        "--host", args.host,
        "--port", str(args.port),
        "--refresh-rate", str(args.viz_refresh_rate),
    ]

    processes = []

    try:
        print("=== 对抗训练启动器 ===")
        print(f"预训练Checkpoint: {args.pretrained_ckpt}")
        print(f"训练轮数: {args.updates}")
        print()

        # 启动训练进程
        print("启动训练进程...")
        train_process = subprocess.Popen(train_args)
        processes.append(train_process)

        # 启动可视化进程
        if not args.no_viz:
            print("启动可视化进程...")
            viz_process = subprocess.Popen(viz_args)
            processes.append(viz_process)
        else:
            print("跳过可视化")

        print()
        print("训练和可视化已启动！")
        print("按 Ctrl+C 停止所有进程")
        print()

        # 等待进程结束
        train_process.wait()

        if not args.no_viz and viz_process.poll() is None:
            print("训练结束，等待可视化窗口关闭...")
            viz_process.wait()

    except KeyboardInterrupt:
        print("\n收到中断信号，正在停止所有进程...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        print("所有进程已停止")


if __name__ == "__main__":
    main()
