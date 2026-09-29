# Linux 服务器训练与 Windows 本机监控操作指南

## 1. 方案边界

CARLA、训练程序、模型和日志全部位于 Linux 服务器。本机只通过 SSH
隧道访问两个只监听服务器回环地址的服务：

- `8080`：CARLA 车载摄像头或跟随车辆的俯视画面；
- `6006`：TensorBoard 指标页面。

网页监控默认关闭，不属于 checkpoint 的科学训练配置。启用后，训练线程
只向容量为 1 的队列提交最新画面；拼图、JPEG 编码、网页客户端和可选落盘
均在后台线程完成。客户端网速慢时会丢弃旧帧，不会等待浏览器，也不会阻塞
CARLA 的同步 `world.tick()`。

正式实验的 `--experiment-contract` 也允许这些明确列出的监控参数，科学训练
参数仍由实验配置固定，源码指纹和严格续训检查保持生效。

俯视模式另外创建一个只用于监控的 RGB 传感器，并在训练 tick 前更新其位置。
它不额外 tick、不等待摄像头回调、不进入策略观测或奖励，但会增加 GPU 渲染和
摄像头位置更新开销；后台编码不代表额外摄像头没有性能成本。

注意：虽然监控参数本身不进入 checkpoint 的科学配置，但该功能修改了训练入口
源码，因此从旧提交严格续训时，现有代码指纹检查会按预期报告源码变化。正式
实验应优先从此分支开启新 run；如确需续训旧 run，应先按项目的代码漂移审核
流程确认差异，不要仅为看到画面而绕过实验协议。

## 2. 服务器取得代码

首次克隆：

```bash
cd /root
git clone <仓库地址> carla-project
cd /root/carla-project
git fetch origin
git switch codex/server-remote-monitor
```

如果仓库已经存在：

```bash
cd /root/carla-project
git fetch origin
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
```

实验输出、checkpoint 和本地日志不要提交到该分支。建议将长期产物写入平台的
持久化目录，并定期用 `rsync` 下载到 Windows。

## 3. 安装新增依赖

进入服务器已经创建好的 `carla-mappo` 环境，只补充项目清单中缺少的包：

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
python -m pip install -r requirements.txt
```

Web 监控新增的直接依赖只有 Pillow。不要在 CARLA 运行用户和训练用户之间混用
Python 环境。

## 4. 启动 CARLA

CARLA 不能以 root 身份运行。沿用已经验证过的 `carla` 用户和 NVIDIA Vulkan
配置：

```bash
mkdir -p /tmp/xdg-runtime-carla
chown carla:carla /tmp/xdg-runtime-carla
chmod 700 /tmp/xdg-runtime-carla

cd /root/carla_runtime/carla/CARLA_0.9.16
runuser -u carla -- env \
  HOME=/home/carla \
  XDG_RUNTIME_DIR=/tmp/xdg-runtime-carla \
  ./CarlaUE4.sh \
  -RenderOffScreen \
  -nosound \
  -quality-level=Low \
  -carla-rpc-port=2000 \
  -graphicsadapter=0
```

如果实际安装目录不同，只调整 `cd` 路径。不要重新指定错误的
`/usr/share/vulkan/icd.d/nvidia_icd.json`；当前机器的 NVIDIA ICD 位于
`/etc/vulkan/icd.d/nvidia_icd.json`，省略 `VK_ICD_FILENAMES` 已验证可用。

## 5. 启动训练和网页画面

保持原训练命令和实验参数不变，仅追加监控参数：

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training

python -m training.train_no_rllib_mappo_robust_v2 \
  <原有训练参数> \
  --host 127.0.0.1 \
  --port 2000 \
  --monitor-mode web \
  --monitor-host 127.0.0.1 \
  --monitor-port 8080 \
  --monitor-fps 5 \
  --monitor-width 960 \
  --monitor-jpeg-quality 65
```

终端出现以下信息表示 HTTP 服务已启动：

```text
Web 训练监控已启动: http://127.0.0.1:8080/
```

### 从浏览器观察俯视画面

在上面的训练命令中追加：

```bash
--monitor-view overhead --monitor-overhead-height 35
```

`--monitor-view vehicle` 是默认的车载摄像头拼图；`overhead` 则显示一张覆盖
全部存活训练车辆的俯视图，摄像头垂直朝下并持续跟随车辆中心。高度以车辆
所在路面为基准，默认至少 35 米；车辆分散时会自动升高，留出取景余量。
观察并线细节可将最低高度改为 25 米，想看更大范围可改为 50 米。
视角随启动参数选择，当前网页没有运行时切换按钮。

`--follow-spectator` 只调整 CARLA 内置观察镜头，不会改变网页视频源，因此
无需为网页俯视模式添加它。俯视画面仍由 RGB 传感器渲染，服务器继续使用
`-RenderOffScreen`，不要启用 `no_rendering_mode`。

每个回合结束后的 reset 会销毁旧的俯视传感器，在下一次训练 step 前重建；
旧传感器迟到的图像会被丢弃。`/status.json` 的 `view` 显示当前视角，
`camera_frame` 是最新俯视图对应的 CARLA 帧号，可能落后于训练的 `step`。
`--monitor-fps` 同时限制网页提交频率（墙钟秒）和俯视传感器采样频率
（模拟秒），所以实际网页更新率还取决于训练推进速度。

已启动的旧训练进程不会因更新文件而切换视角；需要在训练安全保存后，
使用包含新参数的命令重新启动。严格续训的源码指纹限制仍适用，见第 1 节。
不要只为切换画面直接中断尚未保存的 update。

传感器参数说明：[CARLA RGB camera](https://carla.readthedocs.io/en/latest/ref_sensors/#rgb-camera)。

可选地抽样保存监控帧：

```bash
--monitor-record-dir /mnt/experiment_artifacts/monitor_frames
```

记录会消耗磁盘，不建议长训练默认开启。网页监控不需要 `--render`；后者是
本地图形桌面的 pygame 窗口，在无桌面的服务器上应保持关闭。

## 6. 启动 TensorBoard

另开一个服务器终端：

```bash
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
tensorboard \
  --logdir /path/to/current/run/tensorboard \
  --host 127.0.0.1 \
  --port 6006
```

`--logdir` 应指向训练终端打印的 `TensorBoard 日志` 路径，也可以指向多个
run 的上级目录。

## 7. Windows 建立 SSH 隧道

在 Windows PowerShell 中保持以下命令持续运行：

```powershell
ssh -N `
  -L 8080:127.0.0.1:8080 `
  -L 6006:127.0.0.1:6006 `
  -p <矩池云SSH端口> `
  root@<矩池云SSH主机>
```

随后在 Windows 浏览器打开：

- CARLA 画面：`http://127.0.0.1:8080/`
- TensorBoard：`http://127.0.0.1:6006/`

服务只绑定 `127.0.0.1`，无需向公网开放 8080、6006 或 CARLA 2000 端口。
SSH 断开只会让本机暂时看不到网页，不会使服务器训练停止。训练本身应放在
`tmux`、`screen` 或平台的后台任务中运行。

## 8. 带宽和性能建议

默认 `960 px / 5 FPS / JPEG 65` 面向普通宽带和远程调试。网络较差时依次调整：

```bash
--monitor-fps 2 --monitor-width 640 --monitor-jpeg-quality 50
```

网络较好并且需要观察细节时，可提高到 8 FPS，但不建议把网页监控当成无损
视频录制。实际训练判断应以 TensorBoard、结构化 JSONL 和 checkpoint 为准。

## 9. 自检与故障定位

服务器本机检查：

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/status.json
ss -lntp | grep -E ':2000|:6006|:8080'
```

常见现象：

- `/healthz` 正常但无画面：等待环境 reset 后摄像头产生第一帧；
- Windows 无法打开但服务器 `curl` 正常：检查 SSH 隧道命令和 SSH 会话；
- 8080 被占用：训练参数和 Windows `-L` 两端同步改为其他端口；
- 画面卡顿但训练正常：降低 FPS、宽度和 JPEG 质量；
- 训练终端提示一次“Web 监控提交失败”：监控异常被隔离，训练仍会继续。
- `state=camera_error`：俯视传感器创建或更新失败，训练继续，下一次 reset
  会重试；检查训练终端的 `[monitor]` 信息和 CARLA 渲染配置。

状态接口会显示当前 run、update、episode、step、scenario、各 agent 奖励、连接
客户端数和监控丢帧数，可用于确认看到的是否是当前训练。
