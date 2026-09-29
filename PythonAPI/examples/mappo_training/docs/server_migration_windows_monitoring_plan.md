# CARLA 0.9.16 + MAPPO 迁移到 Linux 服务器并在 Windows 本机监控：实施计划

> 适用项目：本仓库 `PythonAPI/examples/mappo_training/`
> 目标环境：矩池云或其他带 NVIDIA GPU 的 Linux 服务器
> 本机环境：Windows
> 推荐方式：服务器离屏运行 CARLA 和训练，浏览器通过 SSH 隧道查看画面与 TensorBoard
> 文档状态：实施计划；尚未在代码中加入 Web 直播模块

## 1. 最终目标与推荐结论

最终系统应当是：

```text
Linux GPU 服务器（训练不依赖本机网络）
├── CARLA 0.9.16 Docker 容器
│   └── -RenderOffScreen：无桌面窗口，但 RGB 传感器正常输出
├── Python 3.10 虚拟环境
│   └── MAPPO 训练进程：唯一负责 world.tick()
├── Web 监控线程
│   ├── 读取训练环境已有的 camera_images
│   ├── 只保留最新一帧，慢网时主动丢帧
│   └── 127.0.0.1:8080 提供 MJPEG/状态页面
├── TensorBoard
│   └── 127.0.0.1:6006
└── 持久化数据盘
    ├── checkpoints
    ├── metrics.jsonl / episodes.jsonl / tb
    └── videos
              │
              │ SSH 加密隧道，仅传监控数据
              ▼
Windows 本机
├── http://127.0.0.1:8080  实时画面
├── http://127.0.0.1:6006  TensorBoard
└── Git + SCP/rsync           同步代码与实验产物
```

核心原则：

1. CARLA 与训练进程必须位于同一台服务器或同一高速内网，训练控制链路不经过本机。
2. 本机断网只会让直播和 TensorBoard 暂时不可见，不得阻塞或中断训练。
3. 同步模式下只能由训练进程推进 `world.tick()`；Web 监控、浏览器和其他客户端都不能 tick、切图或修改 world settings。
4. 代码通过 Git 同步；checkpoint、日志、视频通过持久化数据盘和 SCP/rsync 同步，绝不提交到 Git。
5. 正式实验固定 Git commit、配置文件哈希和依赖锁，不在运行途中 `git pull`。

## 2. 网速会影响什么

### 2.1 结论

采用上述架构后，公网网速不会影响：

- CARLA 与训练脚本之间的 RPC；
- `world.tick()`；
- rollout 采集；
- PPO 更新；
- checkpoint 保存；
- 服务器上的 TensorBoard 日志写入。

公网网速只影响：

- Windows 浏览器看到画面的清晰度、帧率和延迟；
- 打开 TensorBoard 页面的速度；
- 下载 checkpoint、视频和日志的时间。

正确实现中，浏览器关闭、SSH 隧道断开或本地关机都不会终止训练。

### 2.2 为什么不把原始 CARLA 图像直接传到 Windows

当前环境每辆车的 RGB 摄像头为 `640 × 480 × 3`，仿真步长为 0.1 秒，即理论上每秒 10 个仿真帧。以 2 个智能体估算：

```text
640 × 480 × 3 字节 × 10 FPS × 2 ≈ 18.4 MB/s ≈ 147 Mbps
```

CARLA 原始缓冲区还可能包含 Alpha 通道，按 4 字节计算约为 197 Mbps。再考虑协议开销和多个传感器，普通家庭上行/下行和跨地域链路并不适合直接传原始帧。

服务器先拼图、缩放和 JPEG/H.264 压缩后，带宽会明显降低：

| 监控方式 | 建议配置 | 典型带宽估计 | 对训练的影响 | 用途 |
|---|---:|---:|---|---|
| TensorBoard | 标量为主 | 通常低于 0.5 Mbps | 几乎无 | 始终开启 |
| MJPEG 省流 | 960×360、2 FPS、质量 55 | 约 0.3–1.5 Mbps | 很低 | 弱网 |
| MJPEG 默认 | 960×480、5 FPS、质量 65 | 约 1–4 Mbps | 低 | 推荐 |
| MJPEG 清晰 | 1280×720、8 FPS、质量 75 | 约 4–12 Mbps | 中等 | 短时调试 |
| H.264 | 720p、5–10 FPS | 约 0.5–3 Mbps | 需要编码器 | 后续优化 |
| VNC | 动态画面 | 常见 2–15 Mbps，波动较大 | 图形桌面有额外开销 | 临时排障 |

这些数值取决于场景复杂度、JPEG 质量和运动幅度，应以服务器网卡监控结果为准。

### 2.3 防止弱网拖慢训练的硬性设计

Web 监控模块必须满足：

- 使用独立后台线程；
- 训练线程只调用一次非阻塞 `submit()`；
- 队列容量固定为 1，只保存最新画面；
- 新画面到来时覆盖旧画面，不等待浏览器消费；
- 没有浏览器连接时不做或少做 JPEG 编码；
- 客户端断开时捕获异常，不传播到训练线程；
- 监控线程绝不调用 CARLA API 的 `tick/apply_settings/load_world/reload_world`；
- 默认绑定 `127.0.0.1`，仅通过 SSH 隧道访问；
- Web 模块异常只能记录警告，不得使训练失败。

## 3. 当前项目现状与代码修改量

### 3.1 已有能力

当前代码已经具备大部分基础设施：

- `environment/carla_multi_agent_env.py`
  - `_attach_camera()` 为每个车辆创建 640×480 RGB 摄像头；
  - `_parse_image()` 已将图像转成 NumPy 数组并保存；
  - `get_camera_images()` 可以取得 `{agent_id: image}`；
  - 训练环境使用同步模式并负责 `world.tick()`。
- `training/train_no_rllib_mappo_robust_v2.py`
  - 已有 `--host`、`--port`、`--save-dir`、`--save-interval`；
  - 已有 TensorBoard/JSONL 指标记录；
  - 每个完整 update 先保存 `latest.pt`，再提交指标；
  - 支持严格 `--resume latest`；
  - 支持停止请求文件，在完整 update 边界安全停止；
  - `--render` 当前只处理 Pygame 事件，训练循环中没有实际调用 `visualizer.render()`，不能作为服务器直播方案。

### 3.2 推荐改动范围

预计生产级改动为 **200–350 行代码**，不包含测试文档：

| 文件 | 操作 | 预计改动 | 内容 |
|---|---|---:|---|
| `visualization/web_training_monitor.py` | 新增 | 120–200 行 | 最新帧缓冲、拼图、JPEG、HTTP 路由、健康检查、关闭逻辑 |
| `training/train_no_rllib_mappo_robust_v2.py` | 修改 | 35–70 行 | CLI 参数、启动、按频率提交帧、`finally` 关闭 |
| `requirements.txt` | 修改 | 1–3 行 | 明确 Pillow；若用标准库 HTTP 则无需 Flask/FastAPI |
| `tests/test_web_training_monitor.py` | 新增 | 50–100 行 | 无客户端不阻塞、慢客户端丢帧、端口关闭、拼图测试 |
| 启动脚本/示例配置 | 新增 | 30–80 行 | Linux 启动和 Windows 隧道模板 |

第一版不需要修改奖励、策略、观测、动作、CARLA 同步逻辑或 checkpoint schema。

### 3.3 建议增加的命令行参数

```text
--monitor-mode off|web|record|web+record   默认 off
--monitor-host 127.0.0.1
--monitor-port 8080
--monitor-fps 5
--monitor-width 960
--monitor-jpeg-quality 65
--monitor-overlay                         显示 RunID/update/episode/step/reward
--monitor-record-dir <path>               可选录像目录
```

正式实验默认 `off`，显式启用才改变监控行为。监控参数不应进入决定训练语义的不可变参数集合，因为它们不能影响采样和更新结果。

## 4. 迁移前的版本与资源决策

### 4.1 推荐服务器镜像

建议选择：

- Ubuntu 22.04；
- Python 3.10；
- NVIDIA 驱动支持 CUDA 12.x；
- 支持 Docker 和 NVIDIA Container Toolkit；
- 至少 16 GB 内存，建议 32 GB；
- GPU 显存至少 8 GB，正式训练建议 12 GB 或更多；
- 有明确的持久化数据盘。

不要优先选择截图中 Python 3.13 的热门镜像。当前本地环境是 Python 3.10，CARLA 0.9.16 的 Linux Python 客户端与 3.10 组合更稳妥。

### 4.2 当前本地关键版本基线

迁移前已观察到的本地版本：

```text
Python       3.10.8
CARLA        0.9.16
PyTorch      2.9.1+cu126
NumPy        2.2.6
Pygame       2.6.1
TensorBoard  2.20.0
Gymnasium    1.1.1
Ray          2.53.0
Pandas       2.3.3
SciPy        1.15.3
Shapely      2.1.2
PyYAML       6.0.3
Pillow       12.1.0
```

注意：仓库 `PythonAPI/carla/dist/` 中的 `win_amd64.whl` 不能复制到 Linux 使用。Linux 应安装与 Python ABI 对应的 CARLA 0.9.16 Linux wheel，或直接使用 PyPI 的 `carla==0.9.16`。

### 4.3 GPU 分配

单 GPU 时，CARLA 和 PyTorch 会竞争同一张卡。先使用低画质、低帧率监控，并观察：

```bash
nvidia-smi
nvidia-smi dmon
```

如果服务器有两张 GPU，优先：

```text
GPU 0：CARLA 渲染
GPU 1：PyTorch 训练
```

双 GPU 能否使用取决于实例规格和平台对 Docker GPU 的暴露方式。第一轮迁移不要同时追求多 CARLA 实例并行，先完成单实例等价性验收。

### 4.4 实际服务器审计记录（2026-09-25）

已登录的矩池云实例实测如下：

| 项目 | 实测值 | 结论 |
|---|---|---|
| 主机提示符 | `root@Z9aDad` | 当前以 root 登录 |
| 操作系统 | Ubuntu 20.04.6 LTS (Focal) | 不是原计划的 22.04；原生 CARLA 0.9.16 路线仍可评估 |
| Python | 3.10.13 | 满足 CARLA Python 客户端和本项目要求 |
| 当前 Conda 环境 | `/root/miniconda3/envs/myconda` | 平台预装环境，不应直接改造成正式实验环境 |
| GPU | NVIDIA GeForce RTX 4090 | 性能满足单实例 CARLA + MAPPO |
| 显存 | 24564 MiB | 约 24 GB，满足当前训练规模 |
| NVIDIA 驱动 | 580.126.20 | 驱动较新 |
| `nvidia-smi` CUDA 上限 | 13.0 | 表示驱动支持上限，不等于 PyTorch 必须安装 CUDA 13 |
| Docker | `command not found` | 当前镜像未提供 Docker CLI/daemon |
| 根文件系统 | `overlay`，200 GB，约 200 GB 可用 | 很可能处于平台容器内；是否随实例释放而丢失必须确认 |
| `/mnt` | `fuse.fx`，总计 5 GB | 平台网盘空间太小，不能放 CARLA 引擎/镜像和完整长期运行产物 |

当前 `myconda` 中与项目相关的已安装版本：

```text
Python          3.10.13
PyTorch         2.1.2+cu121
TorchVision     0.16.2+cu121
TorchAudio      2.1.2+cu121
NumPy           1.26.4
TensorBoard     2.16.2
Pandas          2.2.1
SciPy           1.12.0
Pillow          10.2.0
OpenCV          4.9.0.80
Matplotlib      3.8.3
scikit-learn    1.4.1.post1
```

与本地基线相比，该环境的 PyTorch、NumPy、TensorBoard、Pandas、SciPy 和 Pillow 均较旧，并且审计输出中没有发现 `carla`、`pygame`、`PyYAML`、`shapely`、`gymnasium` 或 `ray`。`ray` 仅 RLlib 路线需要，当前非 RLlib 训练可不安装。

审计结论：

1. GPU 和 Python 合格，不需要更换实例是因为算力不足。
2. 当前最大的阻塞项是 Docker 不存在和存储持久化策略不明确。
3. 根目录为 `overlay`，说明直接 `apt install docker` 很可能遇到嵌套容器、daemon、cgroup、网络或 GPU runtime 权限问题；在矩池云确认该实例支持 Docker 之前，不执行安装。
4. `/mnt` 只有 5 GB，不在其中安装 Conda 环境、CARLA 引擎或 Docker 数据目录。它暂时只适合保存关键 checkpoint、配置、环境清单和少量日志。
5. 不污染 `myconda`。确定 Docker/原生 CARLA 路线后，新建独立 `carla-mappo` 环境；环境可临时放在 200 GB 根盘，同时把可重建清单和关键实验结果同步到 `/mnt` 或外部存储。
6. NVIDIA 驱动显示 CUDA 13.0，并不妨碍使用 PyTorch `cu121` 或 `cu126` wheel；仍需单独运行 `torch.cuda.is_available()` 和一次 CUDA tensor 测试。

当前路线决策：

```text
优先路线 A：联系/查看矩池云实例说明，确认是否支持 Docker。
    ├── 支持：启用平台推荐的 Docker 方式，继续使用官方 carlasim/carla:0.9.16。
    └── 不支持：不要在 overlay 容器中强装 Docker，改走 CARLA 0.9.16 Linux 原生包。
```

在作出路线决策前，需要补充执行：

```bash
echo "===== TORCH CUDA ====="
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"

echo "===== MEMORY / CPU ====="
free -h
nproc

echo "===== STORAGE DETAIL ====="
findmnt -T /
findmnt -T /mnt
du -sh /root/miniconda3/envs/myconda

echo "===== REQUIRED PACKAGES ====="
python -c "import importlib.util as u; names=['carla','pygame','yaml','shapely','gymnasium','ray']; print({n: bool(u.find_spec(n)) for n in names})"
```

同时必须在矩池云控制台或支持文档中确认两个问题：

- 当前实例是否属于“支持 Docker”的实例，是否允许启动 Docker daemon 和使用 NVIDIA runtime；
- 释放/关机/保存环境后，200 GB overlay 根盘和 5 GB `/mnt` 各自是否保留。

## 5. 数据和目录规划

先在服务器执行：

```bash
df -h
pwd
```

确认矩池云实例中哪个目录是持久化数据盘。下面以 `/mnt/data` 为例；如果平台实际路径不同，全部替换。2026-09-25 实测实例只有 5 GB 的 `/mnt` 网盘，当前示例路径不能直接照搬，必须先购买/扩容持久化盘，或采用“根盘运行 + `/mnt` 保存关键证据 + 外部定期归档”的过渡方案。

推荐目录：

```text
/mnt/data/carla_project/
├── src/                         Git 工作区，只放代码和小配置
├── venv/                        Linux Python 虚拟环境
├── runs/
│   └── <run_id>/
│       ├── latest.pt
│       ├── interrupted.pt
│       ├── checkpoints/
│       ├── manifest.json
│       ├── metrics.jsonl
│       ├── episodes.jsonl
│       ├── tb/
│       └── videos/
├── datasets/                    大型数据和场景资产
└── backups/                     关键实验归档
```

必须确认：关闭或释放实例后 `/mnt/data/carla_project/runs` 是否保留。不要把唯一 checkpoint 放在容器可写层、`/tmp` 或会随实例释放的系统盘。

对于当前实测实例，建议在持久化规则确认前暂定：

```text
/root/carla_runtime/              运行期大文件：CARLA、源码副本、独立 Conda 环境
/mnt/carla_evidence/              关键小文件：配置、环境清单、manifest、周期 checkpoint
外部 Git/本机/对象存储            代码和完整实验归档
```

这只是过渡布局。若根盘随实例释放而丢失，必须提高 checkpoint 外传频率；若单个 checkpoint 较大，5 GB `/mnt` 也必须扩容。

## 6. 代码与实验结果的同步策略

### 6.1 代码：Git 是唯一来源

本地提交代码：

```powershell
cd D:\study\yan2\CARLA_0.9.16
git status
git add <明确选择的文件>
git commit -m "Add server-side training monitor"
git push origin <branch>
```

服务器首次获取：

```bash
mkdir -p /mnt/data/carla_project
cd /mnt/data/carla_project
git clone https://github.com/hy1224hy/CARLA_0.9.16.git src
cd src
git checkout <固定分支或commit>
git rev-parse HEAD
```

正式训练建议直接 checkout 固定 commit：

```bash
git checkout <commit_sha>
git status --short
```

若 `git status --short` 非空，不启动正式实验。训练过程中不执行 `git pull`；需要修改时新开分支或停止并创建新 RunID。

### 6.2 实验产物：不进入 Git

以下文件通过数据盘、SCP、rsync、WinSCP 或对象存储同步：

- `latest.pt`、`interrupted.pt`；
- `checkpoints/*.pt`；
- `manifest.json`；
- `metrics.jsonl`、`episodes.jsonl`；
- TensorBoard `tb/`；
- 评估报告与 MP4。

优先下载不可变的周期 checkpoint，而不是训练正在替换的 `latest.pt`。如果下载 `latest.pt`，同时下载 `manifest.json` 并核对 SHA256。

Windows SCP 示例：

```powershell
scp -P <SSH端口> -r root@<矩池云主机>:/mnt/data/carla_project/runs/<run_id> D:\study\yan2\server_runs\
```

数据量大时，建议在 WSL/Git Bash 中使用支持断点续传的 `rsync`，或者使用矩池云提供的数据盘/网盘工具。不要每分钟下载整个 run 目录；在线查看用 TensorBoard，训练完成后再归档。

### 6.3 Codex 的工作方式

本地 Codex 继续以本地 Git 工作区为主：

1. Codex 在本地修改并测试代码；
2. 提交到独立分支并 push；
3. 服务器 checkout 对应 commit；
4. 训练结果通过远程 TensorBoard查看，通过 SCP/rsync 回传。

本地 Codex 可以通过终端执行明确的 `ssh` 命令读取服务器状态，但本地工作区不会自动变成远程文件系统。不要同时在本地和服务器直接编辑同一个未提交文件，否则容易产生双向覆盖。

## 7. 服务器初始化

### 7.1 基础检查

登录后执行：

```bash
uname -a
cat /etc/os-release
python3 --version
nvidia-smi
docker --version
docker info
df -h
free -h
```

验收标准：

- Ubuntu 22.04；
- Python 3.10；
- `nvidia-smi` 正常；
- Docker 可运行；
- Docker 能看到 NVIDIA GPU；
- 数据盘空间足够且确认持久化。

测试 Docker GPU：

```bash
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi
```

若失败，先解决 NVIDIA Container Toolkit 或实例权限问题，不要继续安装 CARLA。

#### 当前实测实例的 Docker 决策门

2026-09-25 实测为 `docker: command not found`，且 `/` 是 overlay 文件系统。此时不要直接执行通用 Ubuntu 教程中的 `apt install docker.io`。先在矩池云控制台确认实例类型：

- 如果平台明确标注支持 Docker，按照矩池云该实例类型的启用方式安装/启动 Docker，并验证 `docker run --rm --gpus all ... nvidia-smi`；
- 如果平台不支持 Docker-in-Docker，选择新的 Docker 实例，或者执行第 8.4 节的 Linux 原生 CARLA 路线；
- 不把 Docker Root Dir 放到当前 5 GB `/mnt`。

### 7.2 创建 Linux Python 环境

```bash
python3.10 -m venv /mnt/data/carla_project/venv
source /mnt/data/carla_project/venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

不要复制 Windows 的 `venv/`。虚拟环境含平台相关二进制，必须在 Linux 上重新创建。

对于 2026-09-25 实测实例，建议使用现有 Miniconda 新建环境，而不是修改平台预装的 `myconda`：

```bash
conda create -n carla-mappo python=3.10 pip -y
conda activate carla-mappo
python --version
python -m pip install --upgrade pip setuptools wheel
```

该环境默认位于 `/root/miniconda3/envs/carla-mappo`，会使用 200 GB 根盘。由于根盘持久化尚未确认，应把环境导出文件放到 `/mnt`：

```bash
mkdir -p /mnt/carla_evidence/environment
conda env export --no-builds > /mnt/carla_evidence/environment/carla-mappo.yml
python -m pip freeze > /mnt/carla_evidence/environment/requirements-linux.txt
```

在 Docker/原生 CARLA 路线和根盘持久化规则确认前，可以创建这个空环境，但不要立即批量升级为正式训练依赖。

当前 `requirements.txt` 的版本约束较宽，第一轮迁移建议显式安装并验证：

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.9.1
python -m pip install \
  carla==0.9.16 \
  numpy==2.2.6 \
  pygame==2.6.1 \
  tensorboard==2.20.0 \
  Pillow==12.1.0 \
  PyYAML==6.0.3 \
  scipy==1.15.3 \
  pandas==2.3.3 \
  shapely==2.1.2
```

只有实际使用 RLlib 脚本时再安装 `ray[rllib]==2.53.0`，避免为当前非 RLlib 训练增加依赖和排障面。

如果服务器 CUDA/驱动不支持 `cu126`，应根据 `nvidia-smi` 和 PyTorch 官方安装矩阵选择兼容 wheel；无需让系统 CUDA Toolkit 与本地完全相同，关键是 NVIDIA 驱动满足 PyTorch wheel 和 CARLA 容器要求。

安装后记录 Linux 锁文件：

```bash
python -m pip check
python -m pip freeze > /mnt/data/carla_project/runs/environment-linux.txt
python -c "import torch, carla, numpy; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(numpy.__version__)"
```

不要直接用 Windows `pip freeze` 作为 Linux 安装清单，因为其中可能含 Windows 专属 wheel。

## 8. 启动并验证 CARLA 0.9.16

### 8.1 拉取官方容器

```bash
docker pull carlasim/carla:0.9.16
```

### 8.2 离屏启动

```bash
docker run -d \
  --name carla0916 \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --env NVIDIA_VISIBLE_DEVICES=all \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  carlasim/carla:0.9.16 \
  bash CarlaUE4.sh \
    -RenderOffScreen \
    -nosound \
    -quality-level=Low \
    -carla-rpc-port=2000
```

部分 Docker 版本不允许同时写 `--runtime=nvidia` 和 `--gpus all`。如命令报参数冲突，保留平台支持的那一种；CARLA 官方示例使用 `--runtime=nvidia` 和 NVIDIA 环境变量。

检查日志：

```bash
docker ps
docker logs --tail 100 carla0916
ss -lntp | grep 2000
```

`-RenderOffScreen` 与 CARLA 的 `no_rendering_mode` 不同：前者仍生成摄像头图像；后者会使摄像头和 GPU 传感器返回空数据。本项目要远程看 RGB，不能开启 `no_rendering_mode=True`。

### 8.3 Python 连通性测试

激活虚拟环境后：

```bash
python -c "import carla; c=carla.Client('127.0.0.1',2000); c.set_timeout(10); w=c.get_world(); print(w.get_map().name)"
```

然后确认加载 Town05 和传感器可用。正式训练脚本会加载地图，因此连通性测试不要与正在运行的训练同时执行切图操作。

### 8.4 无 Docker 时的 Linux 原生 CARLA 备选路线

只有矩池云确认当前实例不支持 Docker，而且不准备更换实例时，才使用本路线：

1. 从 CARLA 官方发布页获取 **Linux 版 CARLA 0.9.16 预编译包**；
2. 解压到根盘，例如 `/root/carla_runtime/CARLA_0.9.16/`，不要放入仅 5 GB 的 `/mnt`；
3. 使用 Linux 包自带的 `CarlaUE4.sh`，不能上传 Windows 的 `CarlaUE4.exe`；
4. 仍采用离屏启动：

```bash
cd /root/carla_runtime/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low -carla-rpc-port=2000
```

5. 若缺少 Vulkan/SDL/系统动态库，只安装报错明确指出的运行库，不在未确认前升级系统发行版或显卡驱动；
6. CARLA 大文件是可重新下载的运行资产，关键代码、配置、checkpoint 和环境清单必须另行持久化。

Ubuntu 20.04 与 Python 3.10 不自动构成阻塞；真正的验收标准是：`CarlaUE4.sh` 能使用 RTX 4090 离屏启动、Python `carla==0.9.16` 能连接、RGB 摄像头能返回非空帧。

## 9. Web 直播模块的实施设计

### 9.1 新模块职责

新增：

```text
visualization/web_training_monitor.py
```

建议接口：

```python
monitor = WebTrainingMonitor(
    host="127.0.0.1",
    port=8080,
    fps=5,
    width=960,
    jpeg_quality=65,
)
monitor.start()

# 每个 env.step() 后调用；必须立即返回
monitor.submit(
    images=env.get_camera_images(),
    metadata={"run_id": ..., "update": ..., "episode": ..., "step": ...},
)

monitor.close()
```

HTTP 路由：

```text
GET /              简单监控页面
GET /stream.mjpg   实时画面
GET /status.json   RunID、update、episode、step、最近帧时间
GET /healthz       进程健康检查
```

第一版建议只使用 Python 标准库 `ThreadingHTTPServer` 和 Pillow：

- 依赖少；
- 不需要 Flask/FastAPI/uvicorn；
- 足够支持单用户通过 SSH 隧道查看；
- JPEG 在 CPU 上编码，避免和 CARLA/PyTorch 抢 GPU。

### 9.2 接入训练循环

修改 `training/train_no_rllib_mappo_robust_v2.py`：

1. parser 增加监控参数；
2. 环境成功初始化后创建 monitor；
3. `env.step()` 后取得 `env.get_camera_images()`；
4. 按墙钟时间限制 FPS，而不是按每个仿真 step 都编码；
5. 将画面和少量元数据提交给最新帧缓冲；
6. 在 `finally` 中先关闭 monitor，再关闭环境；
7. monitor 的任何异常只打印警告，不能改变 checkpoint 和实验状态。

不要让 Web 线程直接访问可变的 CARLA actor，也不要让它调用 `world.wait_for_tick()`。它只消费训练线程已经拿到的 NumPy 图像快照。

### 9.3 画面布局

两智能体默认横向拼接：

```text
┌──────────────────────┬──────────────────────┐
│ Agent 0 Camera       │ Agent 1 Camera       │
│ speed / reward       │ speed / reward       │
└──────────────────────┴──────────────────────┘
 RunID | update | episode | step | server time | stream FPS
```

三辆或四辆车时自动使用 2×2 网格。车辆到达后摄像头被销毁，页面对应格子显示 `inactive`，不能因为字典缺图而报错。

### 9.4 与训练确定性的关系

直播代码不能参与：

- observation 构造；
- reward 计算；
- action 采样；
- episode 终止；
- scenario 采样；
- RNG 状态；
- checkpoint 内容。

为验证没有引入训练语义变化，应使用固定 seed 分别执行监控关闭/开启的短 smoke run，对比：

- 场景序列；
- 第一个 update 的 episode 数和步数；
- reward residual；
- checkpoint 中训练配置和 RNG 字段；
- 更新是否正常完成。

耗时可能略有变化，但科学结果不应因浏览器是否连接而改变。

## 10. 启动训练

所有训练命令从以下目录以模块方式运行：

```bash
cd /mnt/data/carla_project/src/PythonAPI/examples/mappo_training
source /mnt/data/carla_project/venv/bin/activate
```

### 10.1 先执行 smoke run

下面只是基础连通性模板。正式实验应继续使用项目已有的实验契约和冻结配置：

```bash
python -m training.train_no_rllib_mappo_robust_v2 \
  --host 127.0.0.1 \
  --port 2000 \
  --town Town05 \
  --updates 1 \
  --rollout-episodes 1 \
  --max-steps 100 \
  --run-id server_smoke_001 \
  --save-dir /mnt/data/carla_project/runs/server_smoke_001 \
  --monitor-mode web \
  --monitor-host 127.0.0.1 \
  --monitor-port 8080 \
  --monitor-fps 3 \
  --monitor-width 960 \
  --monitor-jpeg-quality 60
```

### 10.2 正式训练

正式实验应：

- 使用唯一 RunID；
- 使用原有 `--experiment-contract`/`--config-path`；
- 将 `--save-dir` 指向持久化盘；
- 固定 Git commit；
- 保留默认严格 resume，不随意使用 `--allow-resume-code-drift`；
- 不使用当前 Pygame `--render`；
- 只增加不改变训练语义的 `--monitor-*` 参数。

建议在 `tmux` 中运行：

```bash
tmux new -s carla_train
```

启动命令后按 `Ctrl+B`、再按 `D` 脱离。重新连接：

```bash
tmux attach -t carla_train
```

`tmux` 只能防止 SSH 断开导致 shell 退出，不能防止实例被平台回收。真正的恢复依赖持久化 checkpoint。

### 10.3 安全停止与续训

优先使用训练脚本已有的停止请求文件，使训练在完整 update 边界停下：

```bash
touch /mnt/data/carla_project/runs/<run_id>/STOP_REQUESTED
```

前提是启动命令包含：

```text
--stop-request-file /mnt/data/carla_project/runs/<run_id>/STOP_REQUESTED
```

续训使用项目现有严格协议：

```text
--resume latest
```

并保持相同 `--run-id`、`--save-dir`、seed、训练配置和代码版本。当前代码会检查 Git SHA/关键源码指纹；不要把 `--allow-resume-code-drift` 当作常规选项。

## 11. 启动 TensorBoard

服务器执行：

```bash
tensorboard \
  --logdir /mnt/data/carla_project/runs \
  --host 127.0.0.1 \
  --port 6006
```

也建议放在独立 `tmux` 会话：

```bash
tmux new -s tensorboard
```

TensorBoard 只监听 loopback，不直接暴露公网。

## 12. Windows 本机配置

### 12.1 SSH 配置

编辑：

```text
C:\Users\<用户名>\.ssh\config
```

示例：

```sshconfig
Host matpool-carla
    HostName <矩池云SSH主机>
    User root
    Port <矩池云SSH端口>
    IdentityFile C:/Users/<用户名>/.ssh/id_ed25519
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

优先使用 SSH 密钥，不把密码写入脚本或仓库。

### 12.2 建立监控隧道

PowerShell 执行：

```powershell
ssh -N `
  -L 8080:127.0.0.1:8080 `
  -L 6006:127.0.0.1:6006 `
  matpool-carla
```

随后访问：

```text
实时画面：http://127.0.0.1:8080
TensorBoard：http://127.0.0.1:6006
```

SSH 窗口没有输出属于正常现象。关闭该窗口只会断开本机监控，不会停止服务器训练。

如果本机端口已被占用，可以改左侧端口：

```powershell
ssh -N -L 18080:127.0.0.1:8080 -L 16006:127.0.0.1:6006 matpool-carla
```

浏览器相应访问 `http://127.0.0.1:18080` 和 `http://127.0.0.1:16006`。

### 12.3 不建议转发 CARLA 端口作为主要画面方案

不建议日常转发 `2000–2002` 并让 Windows 客户端直接订阅传感器：

- 原始传感器带宽高；
- 远端客户端可能被误写成第二个 tick owner；
- 跨公网抖动更容易触发超时；
- 本机还要维护匹配的 CARLA Python 客户端和可视化依赖。

只有调试 CARLA API 时才临时转发，并确保本地脚本完全只读。

## 13. 录像和长期复盘

实时直播用于确认“车是否在动、是否卡住、是否频繁碰撞”；它不是正式实验记录。正式结果应另外保存：

1. 每隔固定 update 使用 deterministic policy 做短评估；
2. 评估画面在服务器编码为 MP4；
3. 文件名包含 RunID、update、seed 和场景；
4. 训练完成后同步到本地；
5. 同时保存评估 JSON 和 checkpoint SHA256。

推荐命名：

```text
videos/<run_id>_u000100_seed42_scenario_xxx.mp4
```

直播允许丢帧；正式评估录像不能依赖浏览器是否连接。

## 14. 验收测试

按顺序执行，上一阶段未通过不要进入下一阶段。

### A. 基础环境

- [ ] `nvidia-smi` 正常；
- [ ] Docker 容器能看到 GPU；
- [ ] 数据盘路径和持久化规则已确认；
- [ ] Linux Python 为 3.10；
- [ ] `pip check` 通过；
- [ ] `torch.cuda.is_available()` 为 `True`；
- [ ] `import carla` 成功且版本为 0.9.16。

### B. CARLA

- [ ] `carla:0.9.16` 容器稳定运行；
- [ ] Python 客户端能连接 127.0.0.1:2000；
- [ ] Town05 能加载；
- [ ] `-RenderOffScreen` 下 RGB 图像非空；
- [ ] 未开启 `no_rendering_mode=True`。

### C. 训练

- [ ] 1 update smoke run 完成；
- [ ] `metrics.jsonl`、`episodes.jsonl`、`tb/` 写入；
- [ ] `latest.pt` 和 `manifest.json` 存在；
- [ ] 周期 checkpoint 能加载；
- [ ] 安全停止在完整 update 边界生效；
- [ ] `--resume latest` 能严格续训。

### D. 本机监控

- [ ] Windows SSH 隧道可建立；
- [ ] 浏览器能打开实时画面；
- [ ] TensorBoard 能持续刷新；
- [ ] 关闭浏览器训练仍继续；
- [ ] 断开 SSH 5 分钟训练仍继续；
- [ ] 弱网时画面降帧而训练 step 时间无明显恶化；
- [ ] Web 监控进程退出后训练仍继续。

### E. 可复现性

- [ ] RunID、Git SHA、配置 SHA、依赖快照已记录；
- [ ] 正式 run 的工作区干净；
- [ ] 监控开启/关闭的 smoke run 训练语义一致；
- [ ] 至少一个 checkpoint 已复制到本地并验证 SHA256；
- [ ] 至少一个评估视频可在 Windows 播放。

## 15. 常见故障和处理

### CARLA 容器立即退出

```bash
docker logs --tail 200 carla0916
nvidia-smi
docker info
```

重点检查 NVIDIA Container Toolkit、驱动、显存和 Docker GPU 权限。

### 相机画面为空

确认：

- 使用了 `-RenderOffScreen`；
- 没有设置 `no_rendering_mode=True`；
- 摄像头 actor 存活；
- CARLA 与 Python 客户端均为 0.9.16；
- GPU 显存没有耗尽。

### 训练卡在 tick

确认没有第二个客户端调用：

```text
world.tick()
world.apply_settings()
client.load_world()
client.reload_world()
```

关闭 ROS Bridge、本地 CARLA 控制脚本和其他可能推进同步世界的程序后再测试。

### 直播卡顿但训练正常

这是允许的。依次降低：

```text
monitor_fps: 5 -> 3 -> 2
monitor_width: 960 -> 720 -> 640
jpeg_quality: 65 -> 55 -> 45
```

不要为了让直播流畅而降低训练可靠性。

### 训练也变慢

比较监控开启/关闭时每个 update 的墙钟耗时，并检查：

- 是否在训练线程执行 JPEG 编码；
- 是否队列大于 1 并发生积压；
- 是否每个仿真 step 都进行拼图；
- 浏览器断开后是否仍持续高频编码；
- CPU 是否满载；
- CARLA 与 PyTorch 是否争抢同一张 GPU。

### SSH 断开后训练消失

说明训练在普通交互 shell 中运行。改用 `tmux`，并确认脚本接收到 SIGHUP 时不会随 SSH 会话退出。

### 实例释放后结果消失

说明结果写到了临时系统盘或容器层。立即把 `--save-dir` 改到经平台确认的持久化目录，并在正式训练前实际做一次“保存—关机—恢复—检查文件”的演练。

## 16. 分阶段实施顺序与工作量

### 阶段 0：冻结当前实验基线（0.5 天）

- 整理当前未提交修改；
- 固定一个迁移分支和 commit；
- 记录当前本地版本与训练命令；
- 确认哪些大文件不能进入 Git。

### 阶段 1：服务器基础环境与 CARLA（0.5–1 天）

- 选择 Ubuntu 22.04/Python 3.10/Docker 镜像；
- 检查 GPU 和持久化盘；
- 启动 CARLA 0.9.16 离屏容器；
- 完成 Python API 连通性和 RGB 帧测试。

### 阶段 2：训练迁移与续训验证（0.5–1 天）

- 创建 Linux venv；
- 安装并锁定依赖；
- 运行 1 update smoke；
- 验证 checkpoint、指标、安全停止和严格 resume。

### 阶段 3：Web 监控代码（0.5–1.5 天）

- 新增异步 MJPEG 模块；
- 接入 robust_v2 训练入口；
- 增加测试；
- 验证浏览器/SSH 断开不会影响训练；
- 做弱网参数测试。

### 阶段 4：正式运行与归档（0.5 天）

- 固定 commit、配置和 RunID；
- `tmux` 启动 CARLA、TensorBoard、训练；
- 建立 Windows 隧道；
- 下载一个周期 checkpoint 并验证；
- 生成一个确定性评估视频。

整体预计 **2–4 个工作日**。如果矩池云镜像已经正确支持 Docker GPU，且只实现基础 MJPEG，通常可压缩到 1–2 天；时间主要消耗在服务器 GPU/Docker 兼容性、依赖复现和断线/续训验收，而不是直播代码本身。

## 17. 不推荐作为主方案的替代路线

### VNC

适合临时看完整桌面和排查 CARLA 启动问题，但长期训练会增加图形桌面开销，链路波动也更明显。可作为备用，不作为正式实验的唯一监控手段。

### ROS Bridge + RViz

当前项目不依赖 ROS，且 ros-bridge 与 CARLA 0.9.16 的版本匹配、Windows/WSL 图形环境和同步模式 passive 配置都会增加复杂度。只有后续研究确实需要 ROS topic/TF 时再引入。

### 本机直接连接远端 CARLA

适合短时只读调试，不适合日常训练监控。它会暴露 CARLA 端口、传输大量原始传感器数据，并增加同步世界被第二客户端误操作的风险。

## 18. 官方参考

- CARLA Docker：<https://carla.readthedocs.io/en/latest/build_docker/>
- CARLA Rendering options：<https://carla.readthedocs.io/en/latest/adv_rendering_options/>
- 矩池云 Docker 实例：<https://matpool.com/supports/doc-use-docker-on-matpool/>
- 矩池云 FAQ（SSH 本地端口转发）：<https://www.matpool.com/supports/reference/faqs/>
- 矩池云快速入门/VNC：<https://www.matpool.com/supports/doc-quick-start/>

## 19. 最终推荐配置

第一版正式采用：

```text
CARLA             0.9.16 Docker，RenderOffScreen，Low quality
Python            3.10 Linux venv
训练              robust_v2，服务器本地连接 127.0.0.1:2000
实时画面          MJPEG，960×480，5 FPS，JPEG quality 65
指标              TensorBoard 6006
安全访问          SSH tunnel，不开放公网 Web/CARLA 端口
运行守护          tmux + 完整 update checkpoint
代码同步          Git 固定 commit
实验产物          持久化盘 + 周期性 SCP/rsync 归档
正式证据          checkpoint + JSONL + TensorBoard + 确定性评估 MP4
```

如果实测公网低于约 2 Mbps 或延迟/丢包明显，将直播调整为 `720 px / 2 FPS / JPEG 55`；训练本身无需调整。若后续需要更低带宽，再将 MJPEG 替换为 H.264/WebRTC，但不建议在第一轮迁移中同时增加这项复杂度。
