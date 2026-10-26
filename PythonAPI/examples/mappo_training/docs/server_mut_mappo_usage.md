# MutMAPPO：Git 同步代码，单独上传模型

继续使用 `/root/carla-project/PythonAPI/examples/mappo_training`，不复制第二套项目、不上传源码ZIP。`training/`、`environment/`、`evaluation/`、`models/`、`utils/`放代码，`configs/mut_mappo/`放模板，`data/mut_mappo/`放候选四车案例。模型放`artifacts/`，生成配置放`configs/mut_mappo/generated/`，二者被Git忽略。

当前实现是20维观测/2维动作、冻结红0/1、训练蓝2/3。蓝方由对应红Actor暖启动，四车Critic和优化器新建；两个蓝团队A/B进行普通或选择性互学习。A/B不是红蓝双方；红方再训练、自动红蓝交替进化、通信及完整PSRO闭环未实现。Mut入口尚未接入网页视频，使用spectator和文件状态；旧8080画面不代表Mut实验。

u370是未独立合格的弱基线，南北方向blocking/cut-in四车案例仍为候选。须先完成真实CARLA几何、训练/恢复及评估检查，不能将离线测试当驾驶或算法增益证据。

## 1. 更新服务器代码

先让原基础训练/评估完成或安全退出。Mut依赖共享环境/PPO扩展，更新后旧计划会拒绝源码漂移；如需严格恢复旧基础实验，保留并使用它原来的源码版本，不改旧plan/hash。

```bash
conda activate carla-mappo
cd /root/carla-project
git status --short
git rev-parse HEAD
git switch codex/server-remote-monitor
git pull --ff-only origin codex/server-remote-monitor
cd PythonAPI/examples/mappo_training
mkdir -p artifacts
```

Git报告本地修改时保留后处理，不强制覆盖。原CARLA安装和Conda环境继续使用。

## 2. Windows只上传`.pt`

实际源模型为`logs/.../periodic/ckpt_update_000370.pt`；用户先前提供的`diagnostics/.../u000370.pt`不存在。Windows PowerShell执行，SSH别名须指向当前服务器。选择尚不存在的目标文件名，不覆盖正在引用的模型：

```powershell
scp -F "C:\Users\hy\.sshconfig" `
  "D:\study\yan2\CARLA_0.9.16\PythonAPI\examples\mappo_training\logs\BD_vNext_V6_scratch_s42_u600_20261020_v1\periodic\ckpt_update_000370.pt" `
  px2-jcy.matpool.com:/root/carla-project/PythonAPI/examples/mappo_training/artifacts/red_weak_u370.pt
```

## 3. 校验上传并生成配置（不训练）

```bash
cd /root/carla-project/PythonAPI/examples/mappo_training
python -m utils.prepare_mut_mappo_server \
  --checkpoint artifacts/red_weak_u370.pt \
  --expected-sha256 f0aba30af9f3048278e31100f3892df79351a32401f6c16439fb3c055b293c60 \
  --output-dir configs/mut_mappo/generated/u370_smoke_v1 \
  --device cuda --seed 42
```

先核对源端SHA，再注册缺失的checksum sidecar；已有不一致sidecar不覆盖。来源、20/2模型和案例全部验证后才输出B0关闭互学习/B1普通互学习/B2选择性互学习配置。默认2轮、12000训练tick上限、1轮warmup，仅用于工程检查。teacher gate和独立评估消耗另记，不能据训练tick推断全部耗时。其他模型需先在源端`Get-FileHash -Algorithm SHA256`，再填写自己的SHA。

输出目录必须全新。生成配置和模型在实验运行期间不可覆盖。代码测试已使用合成模型，不依赖私人权重：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest \
  tests.test_prepare_mut_mappo_server tests.test_mut_mappo_runner
```

## 4. 在screen内串行运行

事先启动CARLA0.9.16（沿用离屏Low启动指令），本机2000端口不能有其他训练/评估推进时钟；环境使用Town05。异常同步状态或残留进程先排查，不绕过独占锁。每一步成功后才执行下一步。

```bash
screen -S mut-u370
conda activate carla-mappo
cd /root/carla-project/PythonAPI/examples/mappo_training
export PYTHONFAULTHANDLER=1

# C0无蓝车、C1普通蓝交通，先量出弱策略自身的失败。
python -u -m evaluation.evaluate_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b0.json \
  --team A --split selection --conditions C0 C1 \
  --output-dir evaluations/u370_c0c1_smoke_v1

# B0第1轮与完整A+B恢复第2轮。
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b0.json \
  --run-dir logs/MUT_U370_B0_smoke_v1 --stop-round 1 --allow-candidate
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b0.json \
  --run-dir logs/MUT_U370_B0_smoke_v1 --resume --stop-round 2 --allow-candidate

# B0通过后再串行检查B1/B2，分别建立新run。
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b1.json \
  --run-dir logs/MUT_U370_B1_smoke_v1 --allow-candidate
python -u -m training.train_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b2.json \
  --run-dir logs/MUT_U370_B2_smoke_v1 --allow-candidate
```

Ctrl+A，松开再按D分离；`screen -r mut-u370`重连。PYTHONFAULTHANDLER用于保留原生崩溃线程栈，不是段错误修复。

## 5. 独立评估、选模与恢复

以下以B2为例，B0/B1替换相应config/checkpoint和全新输出目录：

```bash
python -u -m evaluation.evaluate_mut_mappo \
  --config configs/mut_mappo/generated/u370_smoke_v1/b2.json \
  --checkpoint logs/MUT_U370_B2_smoke_v1/latest.pt \
  --team both --split selection --conditions C0 C1 C2 \
  --output-dir evaluations/u370_b2_selection_smoke_v1 --select-best
python -m utils.mut_mappo_session inspect --run-dir logs/MUT_U370_B2_smoke_v1
```

使用配对C1/C2差值、蓝方合法性和安全成本，而不是将红方自身超时都算作攻击成功。选择性门控可以全0、选模可以无合格候选，不降低门槛制造成功。

异常后先恢复干净CARLA，在同配置/源码的原训练命令中加`--resume`；只承认完整A+B边界。两轮已完成时无需恢复。更长候选先导须在新output-dir明确传`--rounds`、`--warmup-rounds`、`--train-tick-budget`，保持各组同源、同seed、同实际tick预算。勿改已运行配置再冒充严格续训。
