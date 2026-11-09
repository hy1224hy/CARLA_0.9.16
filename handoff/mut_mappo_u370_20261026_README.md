# MutMAPPO：u370 弱红方服务器候选实验包

这是独立运行目录，不覆盖原基础训练仓库。来源为监控分支 f50a29c，加上本地开发树的 MutMAPPO 依赖、环境角色/计时扩展和 PPO 辅助损失接口；不是整个开发分支合并。未连接真实 CARLA 或启动实际训练。

## 固定模型及实验含义

- 实际源文件：`logs/BD_vNext_V6_scratch_s42_u600_20261020_v1/periodic/ckpt_update_000370.pt`。用户给出的 `diagnostics/.../u000370.pt` 不存在，因此改用已保存的同轮周期文件。
- 包内冻结副本：`artifacts/red_weak_u370.pt`，SHA256 `f0aba30af9f3048278e31100f3892df79351a32401f6c16439fb3c055b293c60`。
- 两个红方 Actor 为 20维观测、2维动作、hidden128。checkpoint 的双车 Critic 不直接用于四车蓝方训练；蓝方 Critic/优化器新建。
- 红0/1冻结。蓝2/3由相应红Actor暖启动。两组蓝团队A/B各自学习，并按配置交流冻结教师；A/B不是红/蓝双方，也不是两个独立实验种子。
- B0关闭互学习，B1普通互学习，B2选择性互学习。当前是连续策略 KL 的项目适配，不是声称精确复现离散 Mut-DQN。
- u370仅为未独立合格的弱基线。新四车 blocking/cut-in 场景也仍属候选，真实 reset 会检查几何/重叠/接触；失败必须排查，不得绕过检查。
- 此包没有红方再训练、自动红蓝交替进化、通信模块或完整PSRO闭环。

## 上传与解压

Windows PowerShell（SSH别名必须指向当前服务器）：

```powershell
scp -F "C:\Users\hy\.sshconfig" `
  "D:\study\yan2\CARLA_0.9.16\handoff\mut_mappo_u370_20261026.zip" `
  "D:\study\yan2\CARLA_0.9.16\handoff\mut_mappo_u370_20261026.zip.sha256" `
  px2-jcy.matpool.com:/root/
```

服务器上：

```bash
conda activate carla-mappo
cd /root
sha256sum -c mut_mappo_u370_20261026.zip.sha256
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
destination = Path('/root/carla-mut-mappo-u370')
if destination.exists():
    raise SystemExit('目录已存在，停止解压，勿覆盖运行源码或已有实验')
with ZipFile('/root/mut_mappo_u370_20261026.zip') as archive:
    archive.extractall('/root')
PY
cd /root/carla-mut-mappo-u370
python verify_bundle.py
python -c "import carla, torch, numpy; print('CUDA:', torch.cuda.is_available())"
python -m training.train_mut_mappo --config configs/mut_mappo/server_u370_smoke/b0.json --validate-only
```

最后三条不连接CARLA。代码SHA检查不等于真实驾驶/几何验证。配置为CUDA，若环境的Torch不支持CUDA，先排查环境；不要把此包误当作已经验证GPU可用。

## CARLA 和 screen

仍使用单个本机 CARLA 0.9.16、端口2000。保留以前的CARLA启动方式。不能与原基础训练、其他评估或其他蓝方实验同时推进该世界。程序不会替你终止旧任务。

```bash
# 缺少screen时单独安装：apt-get update && apt-get install -y screen
screen -S mut-u370-smoke
conda activate carla-mappo
cd /root/carla-mut-mappo-u370
export PYTHONFAULTHANDLER=1
```

进入训练screen后，先在没有其他tick拥有者时检查地图；如不是Town05，可在共享锁下加载：

```bash
python - <<'PY'
import carla
from utils.mut_mappo_session import CarlaOwner
with CarlaOwner(purpose='Mut u370 readiness'):
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(120)
    if not client.get_server_version().startswith('0.9.16'):
        raise RuntimeError('需要CARLA 0.9.16')
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError('世界已同步，请检查旧训练/评估或崩溃残留，不自动强制改时钟')
    if world.get_map().name.rsplit('/',1)[-1] != 'Town05':
        world = client.load_world('Town05')
    print(world.get_map().name)
PY
```

## 先测弱红方，再检查训练闭环

以下每一步串行执行，报错就停止。评估输出和新训练目录都必须不存在。

```bash
# C0无蓝车 / C1普通蓝交通：先记录R0本来就会发生的失败。
python -u -m evaluation.evaluate_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b0.json \
  --team A --split selection --conditions C0 C1 \
  --output-dir evaluations/u370_c0c1_smoke

# B0第一轮：四车环境、PPO及完整A+B保存。
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b0.json \
  --run-dir logs/MUT_U370_B0_smoke_s42 \
  --stop-round 1 --allow-candidate

# B0完整恢复第二轮；不是载入红方优化器继续基础训练。
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b0.json \
  --run-dir logs/MUT_U370_B0_smoke_s42 \
  --resume --stop-round 2 --allow-candidate

# C0/C1/C2同案例配对；C2使用完成训练的A/B断点。
python -u -m evaluation.evaluate_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b0.json \
  --checkpoint logs/MUT_U370_B0_smoke_s42/latest.pt \
  --team both --split selection --conditions C0 C1 C2 \
  --output-dir evaluations/u370_b0_selection_smoke --select-best
```

不得因C0中红方本来超时，就把所有C2超时当作蓝攻击成功。比较同案例C1/C2差值和蓝合法性、安全成本；无合格候选时选模应明确失败。

## B1/B2互学习工程检查

B0流程通过后再串行执行。每个新run都有独立模型、优化器和输出。

```bash
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b1.json \
  --run-dir logs/MUT_U370_B1_smoke_s42 --allow-candidate

python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/server_u370_smoke/b2.json \
  --run-dir logs/MUT_U370_B2_smoke_s42 --allow-candidate
```

两者的独立评估命令与B0相同，但替换config、checkpoint和全新output-dir。选择性门控可以合法地全0；不得为制造KL而放宽门槛。

这三组都是2轮工程检查；warmup特意缩为1轮，以便在第二轮检查互学习路径。它们不是正式收敛/论文对照实验。正式先导应另建同源、同seed、同实际训练tick预算和默认教师时钟的独立配置，不能修改这三个已运行配置后冒充严格续训。

## 监控、异常恢复与限制

- Ctrl+A，松开再按D：分离screen；重连 `screen -r mut-u370-smoke`。
- 纯文件状态：`python -m utils.mut_mappo_session inspect --run-dir logs/MUT_U370_B0_smoke_s42`。
- 同一配置和源码下，原生崩溃后先恢复干净CARLA，再用对应训练命令加 `--resume`，从完整A+B安全边界继续。未完整提交的另一半不算完成。
- 此Mut入口尚未接入原训练网页视频。spectator跟随及文件状态不是浏览器8080视频；旧网页若仍在显示原实验，不代表Mut训练状态。
- 此包不会自动启动长训练，也没有替你执行服务器操作。正式实验预算和可靠性需以真实小规模运行结果决定。


## 本包离线检查

65 项 MutMAPPO 检查、41 项共享基础/监控回归检查，共 106 项通过；B0/B1/B2 配置和 C0/C1 评估入口离线通过。没有连接 CARLA、执行真实评估或启动服务器训练。详见 offline_validation.json。
