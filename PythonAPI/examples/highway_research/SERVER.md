# Linux服务器启动

服务器不需要CARLA服务端。这些实验使用highway-env和CPU PyTorch，每个worker设置单线程，最多4组并行。

## 安装与验证

在仓库根目录，Python建议3.10或3.11：

```bash
git switch codex/mappo-experiment-baseline
git pull --ff-only
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r PythonAPI/examples/highwaytest/requirements-server.txt
cd PythonAPI/examples
python -m unittest discover -s highway_research/tests -v
```

该requirements文件为仓库已有的精确版本清单；PyTorch可按服务器环境先安装CPU发行版，再安装其余依赖。不同平台不保证位级一致；正式五种子尽量在同一软件环境完成，检查点会保存运行版本并拒绝不匹配续跑。

## 生成配置，开始核心对比

以下命令从PythonAPI/examples执行，使用激活虚拟环境后的python：

```bash
bash highway_research/scripts/experiment.sh plan
bash highway_research/scripts/experiment.sh preflight
```

**必须在服务器重新生成配置**。Windows本地生成的study.json包含绝对路径，不应复制后直接运行。本次提交仅包含生成器，模型、日志、烟测结果及155个本地配置不上传。

建议在tmux或服务器作业系统里启动，避免SSH断线终止编排：

```bash
tmux new -s research1
bash highway_research/scripts/experiment.sh main
# Ctrl+B再按D离开会话；tmux attach -t research1重新进入
```

同一命令可恢复未完成训练、跳过已完成运行。脚本默认4组并行，也可`STUDY_JOBS=2 bash ... main`。不要同时启动两个指向同一study的编排器。只先跑main，审计行为通过后再启动ablations。

自定义存储目录作为第二参数，所有阶段使用相同路径：

```bash
bash highway_research/scripts/experiment.sh plan /data/experiments/research1_v1
bash highway_research/scripts/experiment.sh main /data/experiments/research1_v1
```

## TensorBoard与评估

在另一个会话、激活同一虚拟环境，从PythonAPI/examples运行：

```bash
bash highway_research/scripts/experiment.sh tensorboard
bash highway_research/scripts/experiment.sh audit-dev
bash highway_research/scripts/experiment.sh report
```

本地电脑建立端口转发：

```bash
ssh -L 6012:127.0.0.1:6012 user@server
```

浏览器访问http://127.0.0.1:6012。测试集使用audit-test，须在超参和协议冻结后执行。调参、扰动、随机执行和回放命令详见README.md与研究报告。

新工具采用跨rollout保留回合的采样协议，不直接读取旧训练包的检查点。旧PPO/Mut-PPO实验结果不受代码改写影响，也不要与新工具运行混算为同一批五种子数据。
