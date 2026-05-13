# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a **CARLA 0.9.16** autonomous driving simulator with a custom **multi-agent adversarial RL (MAPPO)** training project under `PythonAPI/examples/mappo_training/`. The CARLA engine (Unreal-based) serves as a server; all RL code interacts with it via the `carla` Python client library.

## Start the Simulator

```bash
# Windows
./CarlaUE4.exe
# Linux
./CarlaUE4.sh
```

The simulator runs a server on `localhost:2000`. All Python scripts connect to it as clients. Use `-quality-level=Low` for headless training.

## Python Environment

There's a local `venv/` at the repo root. Activate it before running anything:

```bash
source venv/Scripts/activate  # Windows Git Bash
```

Key dependencies: `torch`, `numpy`, `carla==0.9.16`, `pygame`, `tensorboard`, `ray[rllib]` (optional).

## Running Training / Evaluation Scripts

All mappo_training scripts must be run as **modules from the `mappo_training/` directory** (or with the parent on `sys.path`):

```bash
cd PythonAPI/examples/mappo_training

# Train standard MAPPO (2 agents)
python -m training.train_no_rllib_mappo --render

# Train adversarial agent (Agent 2) against frozen pretrained Agents 0 & 1
python -m training.train_adversarial_ppo --pretrained-ckpt <path.pt> --curriculum-learning

# Robustness training with random scenarios
python -m training.train_no_rllib_mappo_robust --render

# Evaluate a checkpoint
python -m evaluation.evaluate_no_rllib_mappo --checkpoint <path.pt>
python -m evaluation.evaluate_adversarial_ppo --checkpoint <path.pt>
python -m evaluation.evaluate_robustness --checkpoint <path.pt>
```

Monitor with `tensorboard --logdir PythonAPI/examples/mappo_training/logs/`.

## Architecture

### Core Environment (`environment/carla_multi_agent_env.py`)

`CarlaMultiAgentEnv` is the central RL environment. Key design points:

- **Synchronous mode**: CARLA runs in synchronous mode with `fixed_delta_seconds=0.1`. Every `step()` calls `world.tick()`.
- **Town02 straight road**: On init, scans waypoints to find the longest straight segment. Agents spawn distributed along it, target is the far end.
- **Observation space**: 19-dim vector per agent = 8 base features (dx, dy, distance, angle_diff, speed, throttle, steer, brake) + 11 ray-cast lidar distances (180° FOV, 10m range).
- **Action space**: 3-dim continuous `[throttle, steer, brake]`, applied via `carla.VehicleControl`. Throttle/brake ∈ [0,1], steer ∈ [-1,1].
- **Sensors per vehicle**: RGB camera (640×480, 110° FOV), collision detector, lane invasion detector (with crosswalk detection treated as collision).
- **Reward function**: progress (+20× meters advanced) + speed bonus (0.1 in 1-25 km/h, penalize stopped/speeding) + centering (+0.5 if within 5cm of lane center) + yaw alignment + collision (-100) + lane invasion (-5) + step penalty (-0.05) + goal success (+100 in a 6m×3.5m rectangular zone).
- **Adversarial agent (Agent 2)**: Gets a modified reward combining down-weighted base driving reward + cutting-in bonus + blocking bonus. The training script additionally subtracts other agents' rewards (zero-sum term).
- **Episode termination**: All agents reach goal, any severe collision, or max_steps exceeded.
- **Custom start positions**: Via `set_custom_start_positions(dict)` or `reset_with_random_scenario()` which loads JSON from `data/training_scenarios/`.

### MAPPO Implementation (`training/train_no_rllib_mappo.py`)

- **Architecture**: CTDE (Centralized Training Decentralized Execution).
- **Actor**: 2-layer MLP (128 hidden, Tanh activation) → mu head + learned log_std. Uses tanh-squashed Gaussian policy for bounded actions.
- **CentralizedCritic**: Takes concatenated global state (6 features per agent: x, y, speed, sin(yaw), cos(yaw), dist_to_goal).
- **PPO**: GAE advantage estimation, clipped objective, entropy bonus, gradient clipping.
- **No policy sharing**: Each agent has its own Actor network.

### Adversarial Training (`training/train_adversarial_ppo.py`)

- Agents 0 & 1 loaded from a pretrained checkpoint and **frozen**.
- Agent 2 trained with PPO. Its reward = `adversarial_coef × (-avg_reward_of_agents_0_1)` plus the environment's adversarial bonus terms.
- Supports **curriculum learning**: warmup phase with only base driving reward, then ramp up the adversarial coefficient.

### Robustness Training (`training/train_no_rllib_mappo_robust.py`)

Extends standard MAPPO with random scenario sampling from `data/training_scenarios/*.json` and curriculum learning over scenario difficulty.

### RLlib Wrapper (`environment/carla_rllib_wrapper.py`)

Wraps `CarlaMultiAgentEnv` as an RLlib `MultiAgentEnv` for use with Ray's MAPPO implementation. Used by `train_rllib.py`.

### Speedup Wrapper (`environment/speedup_env_wrapper.py`)

Wraps the environment to make other vehicles drive faster/more aggressively, used for robustness evaluation.

### Visualization

- `MAPPOVisualizer`: pygame-based renderer for standard training.
- `AdversarialVisualizer`: pygame-based renderer for adversarial training, showing per-agent rewards.

## Key File Map

| File | Purpose |
|------|---------|
| `environment/carla_multi_agent_env.py` | Core env: spawning, sensors, observations, rewards, ray casting |
| `training/train_no_rllib_mappo.py` | Standalone MAPPO training (no RLlib dependency) |
| `training/train_adversarial_ppo.py` | Adversarial agent training against frozen pretrained policies |
| `training/train_no_rllib_mappo_robust.py` | Robustness training with random scenarios |
| `training/train_rllib.py` | RLlib-based MAPPO training |
| `training/pretrain_with_manual_data.py` | Behavior cloning pretraining from human driving data |
| `environment/carla_rllib_wrapper.py` | RLlib MultiAgentEnv wrapper |
| `environment/speedup_env_wrapper.py` | Speedup wrapper for stress-testing |
| `evaluation/evaluate_*.py` | Various evaluation scripts for different training variants |
| `utils/record_manual_control.py` | Human driving data collection tool |
| `data/training_scenarios/` | JSON scenario files for randomized training |
| `data/custom_positions_example.json` | Example custom start position config |
| `utils/mappo_logger.py` | Logging utilities |
| `utils/manual_data_loader.py` | Loader for human driving datasets |
| `visualization/mappo_visualizer.py` | Pygame training visualizer |
| `visualization/visualize_adversarial_training.py` | Adversarial training visualizer |

## Important Conventions

- Action tanh mapping: Networks output in `[-1,1]`. The environment expects `throttle ∈ [0,1], steer ∈ [-1,1], brake ∈ [0,1]`. Conversion happens in `action_tanh_to_env()`.
- Checkpoints save: actor state_dicts, critic state_dict, optimizer state, update count. Format: `ckpt_{update}.pt`.
- All reward logging goes to `logs/` subdirectories, auto-created per training run.
- Custom start positions JSON format: `{"agent0": {"x": ..., "y": ..., "z": 0.5, "yaw": 90.0}, ...}`. Yaw is optional — if omitted, auto-aligns to road direction.
