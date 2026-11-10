# 混合动作消融矩阵实施计划(hybrid ablation matrix)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 今晚在混合动作协议(`highway_hybrid_twin_v1`,commit 2108f25)上启动并跑完 55 runs × 500k 步的消融矩阵(主对比/β/奖励三臂),产出 development 冻结审计与分臂汇总报告。

**Architecture:** 全部为配置与编排层工作——零 frozen 源码改动。研究目录 `highwaytest/runs/hybrid_matrix_20261111_v1/` 内放配置生成、编排器(ThreadPoolExecutor 子进程调度,8 workers,A 臂先跑完再 B+C 合池)、监控、审计与报告脚本,复用 pilot 的编排模式。训练/评估全部走 frozen 的 `python -m highway_hybrid train/evaluate` 入口。

**Tech Stack:** Python 3.10.8(venv `D:\study\yan2\CARLA_0.9.16\venv\Scripts\python.exe`)、unittest、subprocess、ThreadPoolExecutor、highway_hybrid 模块(frozen)、TensorBoard。

**Spec:** `docs/superpowers/specs/2026-11-11-hybrid-ablation-matrix-design.md`(本计划从其论证,执行者两份都读)

## Global Constraints

- Python 用 venv 的 `sys.executable`;所有子进程 cwd = `D:\study\yan2\CARLA_0.9.16\PythonAPI\examples`。
- 全部 55 个配置:`total_steps=500000`、`eval_cases=32`、`warmup_steps=20000`、`ramp_steps=20000`、seeds ∈ {42,43,44,45,46};其余参数与 pilot 冻结值一致。
- A 臂:`L{1,2,3}_{independent,mutual}_s{seed}`,β=0.02(mutual)。B 臂:`L2_beta{0.005,0.08}_s{seed}`(method=mutual)。C 臂:`L2_{no_risk,no_event,no_dense}_s{seed}`(分别 risk_weight=0 / event_weight=0 / effect_weight=0,method=mutual)。
- **禁止修改** frozen 源码集(highway_hybrid/*.py、highwaytest/straight/{config,environment,rewards}.py、highwaytest/training/{ppo,checkpoint,locking}.py、highwaytest/config.py、highway_eval/evaluate.py)。smoke 步骤必须验证 source digest 与 pilot(manifest `eeda4f74…`)一致。
- 审计:development 40 固定场景,`--checkpoint selected`;test 100 场景本晚不跑。
- 研究目录固定为 `highwaytest/runs/hybrid_matrix_20261111_v1/`(study);`runs/` 被 gitignore,control 脚本不提交(与 pilot 先例一致);仅 spec+plan 文档走 git。
- 输出目录必须新目录;训练输出 `study/train/<arm>/<name>`;评估输出 `study/audit_selected/<arm>`。
- Commit 消息末尾加 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。

## Review Focus

1. **双重启动**(编排器重复跑或手动重跑同一 run):run_lock 会抛 "another process holds this run"。编排器必须把锁冲突视为**不可重试**失败(否则合法运行的 run 会被重试撞锁死循环)→ 测试:run_job 读到锁消息时不重试。
2. **残缺 JSONL 末行**(监控进程读到训练进程正在写入的半行):read_latest_metrics 必须容忍 JSONDecodeError 回退上一行,不崩溃不误报 → 测试:末行截断的合成文件。
3. **审计在训练未完成时误跑**:audit_arm 必须 gate 在 all_complete(所有 summary.json status==complete)上 → 测试:一个 run 缺 summary 时抛错。
4. **报告复用映射错位**(β 曲线复用主对比的 β=0/β=0.02,full 复用 L2_mutual):group_rows 的复用表必须按组名精确匹配 → 测试:合成 summary 下 β=0 行来自 `L2_independent_s*`。
5. **配置生成静默偏差**(预算/seed/消融字段写错会让 55×2h 计算作废):check_matrix 必须逐项断言,make_configs 生成后立即 check,不通过则中止启动 → 测试:预算不一致、重复名、eval_cases 改动、消融字段写错各一条被拒。
6. **Windows 控制台编码**:训练子进程必须带 `-X utf8 -u`,日志文件 UTF-8 追加 → 测试:build_command 断言 flag 存在。

---

### Task 1: 研究目录骨架与配置生成校验

**Files:**
- Create: `PythonAPI/examples/highwaytest/runs/hybrid_matrix_20261111_v1/control/matrix_configs.py`
- Create: `PythonAPI/examples/highwaytest/runs/hybrid_matrix_20261111_v1/control/make_configs.py`
- Test: `PythonAPI/examples/highwaytest/runs/hybrid_matrix_20261111_v1/control/tests/test_matrix_configs.py`
- (make_configs 运行时创建 study 下的 configs/{main,beta,reward}/)

**Interfaces:**
- Consumes: `HybridConfig`(from `highway_hybrid.config`)、`dataclasses.replace`。
- Produces:
  - `generate_main(study: Path) -> list[Path]`:subprocess 调 `[sys.executable,'-m','highway_hybrid','plan','--output',str(study/'configs'/'main'),'--steps','500000','--seeds','42','43','44','45','46','--levels','1','2','3','--methods','independent','mutual','--beta','0.02']`(cwd=examples),返回生成的 30 个 config 路径。
  - `generate_beta(study: Path) -> list[Path]`:HybridConfig(level=2, method='mutual', seed, total_steps=500000, warmup_steps=20000, ramp_steps=20000) 经 replace 设 beta∈(0.005,0.08),名字 `L2_beta{beta:g}_s{seed}.json`,写入 `study/configs/beta/`,共 10 个。
  - `generate_reward(study: Path) -> list[Path]`:同基座,replace 设 risk_weight=0.0 → `L2_no_risk_s{seed}`;event_weight=0.0 → `L2_no_event_s{seed}`;effect_weight=0.0 → `L2_no_dense_s{seed}`,写入 `study/configs/reward/`,共 15 个。
  - `check_matrix(study: Path) -> dict`:返回 `{'arms': {'main': n, 'beta': n, 'reward': n}, 'total': n}`;任一断言失败抛 ValueError。断言:①数量 main=30/beta=10/reward=15;②每个 JSON 可经 `HybridConfig.from_dict` 加载(内含 validate);③全部 total_steps==500000、eval_cases==32、warmup_steps==20000、ramp_steps==20000;④seed ∈ {42..46} 且每臂 (level,method,seed) 与 (level,seed) 组合覆盖完整(无重复、无遗漏);⑤跨臂文件名唯一;⑥main 臂 beta==0.02;beta 臂 beta∈{0.005,0.08} 且 level==2、method=='mutual';reward 臂恰有一个 (risk_weight,event_weight,effect_weight)==0 且其余两个==1.0,level==2、method=='mutual'。
  - `make_configs.py` CLI:`python control/make_configs.py --study <path> [--check]`——不存在时生成三臂并打印 `check_matrix` 结果 JSON;已存在时只做 check。

- [ ] **Step 1: 写失败测试**

`tests/test_matrix_configs.py`(unittest,sys.path 插入 study 根;用 tempfile 建临时 study):

```python
def test_generate_beta_names_and_fields(tmp):
    study = Path(tmpdir)
    paths = generate_beta(study)
    assert len(paths) == 10
    names = [p.stem for p in paths]
    assert names == [f'L2_beta{beta:g}_s{s}' for beta in (.005, .08) for s in (42,43,44,45,46)]
    cfg = HybridConfig.from_dict(json.loads(paths[0].read_text(encoding='utf-8')))
    assert cfg.level == 2 and cfg.method == 'mutual' and cfg.beta in (.005, .08)

def test_check_matrix_passes(tmp):  # 生成后 check 返回 {'arms':{'main':30,'beta':10,'reward':15},'total':55}

def test_check_matrix_rejects_budget_mismatch(tmp):  # 改一个 total_steps → ValueError

def test_check_matrix_rejects_duplicate_names(tmp):  # beta 臂复制一个改名为 main 臂已有名 → ValueError

def test_check_matrix_rejects_eval_cases(tmp):  # 改 eval_cases=40 → ValueError

def test_check_matrix_rejects_wrong_ablation(tmp):  # no_risk 同时 event_weight=0 → ValueError
```

- [ ] **Step 2: 跑测试确认失败**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`(cwd=study)
Expected: FAIL(ImportError/AttributeError)

- [ ] **Step 3: 实现 `matrix_configs.py` 与 `make_configs.py`**

生成函数按 Interfaces 签名;check_matrix 用 `HybridConfig.from_dict` 载入后按断言清单逐项检查。make_configs 用 argparse(`--study`, `--check`),输出 `json.dumps(result, indent=2, ensure_ascii=False)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: PASS(6 tests)

- [ ] **Step 5: 对真实 study 生成并 check**

Run: `..\..\..\venv\Scripts\python.exe control/make_configs.py --study .`(cwd=study)
Expected: 输出 `{"arms": {"main": 30, "beta": 10, "reward": 15}, "total": 55}`;`configs/` 下 55 个 JSON 存在。

- [ ] **Step 6: Commit(spec+plan 文档)**

```bash
git add PythonAPI/examples/mappo_training/docs/superpowers/specs/2026-11-11-hybrid-ablation-matrix-design.md PythonAPI/examples/mappo_training/docs/superpowers/plans/2026-11-11-hybrid-ablation-matrix.md
git commit -m "docs: hybrid ablation matrix design and implementation plan"
```

(control/ 在 gitignore 的 runs/ 下,不提交。)

---

### Task 2: 编排器(臂序调度、子进程、断点续跑、自动重试)

**Files:**
- Create: `…/hybrid_matrix_20261111_v1/control/matrix_runner.py`
- Test: `…/hybrid_matrix_20261111_v1/control/tests/test_matrix_runner.py`

**Interfaces:**
- Consumes: Task 1 的 configs 目录布局;`highwaytest/training/locking.run_lock` 的锁语义(OS 文件锁,进程被杀自动释放)。
- Produces:
  - `arm_configs(study: Path, arm: str) -> list[Path]`:排序后的该臂 config 列表。
  - `build_command(python: str, config: Path, run_dir: Path) -> list[str]`:`[python,'-X','utf8','-u','-m','highway_hybrid','train','--config',str(config),'--output',str(run_dir)]`;若 `run_dir/'latest.pt'` 存在则追加 `['--resume',str(run_dir/'latest.pt')]`。
  - `run_job(study, cmd, run_dir, log_path, max_retries: int) -> dict{'run','exit_code','attempts'}`:subprocess.run(cwd=EXAMPLES, stdout/stderr 追加到 log_path);exit_code!=0 时:读 log 尾部,含 `another process holds this run` 则**不重试**直接返回;否则重读 build_command(此时含 --resume)重试,最多 max_retries 次;返回最终 exit_code 与 attempts。
  - `waves(study: Path) -> list[list[tuple[str, Path]]]`:第 1 波 = main 臂全部 (arm,config);第 2 波 = beta+reward 臂全部。
  - `run_study(study: Path, jobs: int, max_retries: int, dry_run: bool=False) -> dict`:每波 ThreadPoolExecutor(max_workers=jobs) 提交 run_job;dry_run 时打印每条 build_command 不执行;写入 `control/pipeline_status.json`(stage: 'training:main' → 'training:beta_reward' → 'complete'/'failed',附时间与 counts)与分臂 `control/batch_status_<arm>.json`;任一波存在最终失败(run 重试后 exit_code!=0)则 stage='failed' 并 SystemExit(1)。
- 常量:`EXAMPLES = Path('D:/study/yan2/CARLA_0.9.16/PythonAPI/examples')`(运行时用 `Path(__file__)` 推导)。

- [ ] **Step 1: 写失败测试**

```python
def test_build_command_has_utf8_and_resume(tmp):
    cmd = build_command('python', cfg, run_dir)          # 无 latest.pt
    assert cmd[1:4] == ['-X', 'utf8', '-u']
    (run_dir/'latest.pt').write_bytes(b'x')
    cmd2 = build_command('python', cfg, run_dir)
    assert '--resume' in cmd2 and cmd2[cmd2.index('--resume')+1] == str(run_dir/'latest.pt')

def test_waves_main_first_then_bc(tmp):
    ws = waves(study)                                    # 合成 3 臂 config
    assert [a for a,_ in ws[0]] == ['main']*30
    assert {a for a,_ in ws[1]} == {'beta','reward'}

def test_run_job_retries_then_succeeds(tmp):             # mock subprocess.run 第一次 returncode=1,第二次 0
    result = run_job(...)
    assert result['attempts'] == 2 and result['exit_code'] == 0

def test_run_job_no_retry_on_lock_message(tmp):          # mock 失败且日志尾含 another process holds this run
    result = run_job(...)
    assert result['attempts'] == 1 and result['exit_code'] != 0
```

- [ ] **Step 2: 跑测试确认失败**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: FAIL

- [ ] **Step 3: 实现 `matrix_runner.py`**

按 Interfaces;run_job 的锁消息检测读 log_path 末尾 2KB(`log_path.read_bytes()[-2048:]`);run_study 用 `ThreadPoolExecutor.map`;dry_run 分支不经线程池直接打印。

- [ ] **Step 4: 跑测试确认通过**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: PASS(全部,含 Task 1 的 6 条)

- [ ] **Step 5: dry-run 核对命令清单**

Run: `..\..\..\venv\Scripts\python.exe control/matrix_runner.py --study . --jobs 8 --max-retries 2 --dry-run`(cwd=study)
Expected: 打印 55 条命令,先 30 条 main、后 25 条 beta+reward;每条 `--output` 指向 `train/<arm>/<name>` 且无目录冲突。

---

### Task 3: 监控进程与 TensorBoard

**Files:**
- Create: `…/hybrid_matrix_20261111_v1/control/matrix_monitor.py`
- Test: `…/hybrid_matrix_20261111_v1/control/tests/test_matrix_monitor.py`

**Interfaces:**
- Produces:
  - `read_latest_metrics(run_dir: Path) -> dict | None`:读 `metrics.jsonl` 最后一行(JSONDecodeError 时回退上一行),取 `train_steps, policy_loss, entropy_acceleration, entropy_lane, effective_beta_acceleration, effective_beta_lane, mean_step_reward, decisions_per_second, clip_fraction`。
  - `read_latest_validation(run_dir: Path) -> dict | None`:读 `validation.jsonl` 最后一行,取每个 team 的 `overall['attack_success_rate']`、`overall['collision_rate']`。
  - `snapshot(study: Path) -> dict`:对 `train/*/*/` 每个 run:名字、arm、steps、完成度 `pct = steps/total`(total 从 config.json 读)、ETA(由 decisions_per_second 折算)、最后 loss、validation 成功率;汇总 `{total, complete, running, failed}`(complete 由 summary.json status=='complete' 判定,failed 由 pipeline_status 判定)。
  - `main()`:`--study --interval 600`:循环 `append(monitor.jsonl, {time_utc, snapshot})` + 重写 `monitor.md`(总进度表+分臂表);`pipeline_status.json` stage=='complete' 或 =='failed' 时写最终快照并退出。
- 启动命令(计划执行时用,不是代码接口):`..\..\..\venv\Scripts\python.exe -m tensorboard.main --logdir train --host 127.0.0.1 --port 6011`(后台,study 根目录)。

- [ ] **Step 1: 写失败测试**

```python
def test_read_latest_metrics_skips_truncated_line(tmp):
    p = run_dir/'metrics.jsonl'; p.write_text('{"train_steps":100}\n{"train_steps":2')  # 末行截断
    assert read_latest_metrics(run_dir)['train_steps'] == 100

def test_snapshot_pct_and_eta(tmp):
    # 合成 run:config total_steps=500000,metrics 末行 train_steps=250000,decisions_per_second=100
    snap = snapshot(study)
    row = snap['runs'][0]
    self.assertEqual(row['pct'], .5)
    self.assertAlmostEqual(row['eta_hours'], 250000/100/3600, places=6)
```

(项目用 unittest;近似断言用 `self.assertAlmostEqual`。)

- [ ] **Step 2: 跑测试确认失败**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: FAIL

- [ ] **Step 3: 实现 `matrix_monitor.py`**

按 Interfaces;snapshot 对缺失文件返回 None 字段不抛错(训练刚开始的 run 可能还没有 validation.jsonl)。

- [ ] **Step 4: 跑测试确认通过**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: PASS

---

### Task 4: 分臂审计与汇总报告

**Files:**
- Create: `…/hybrid_matrix_20261111_v1/control/matrix_audit.py`
- Create: `…/hybrid_matrix_20261111_v1/control/matrix_report.py`
- Test: `…/hybrid_matrix_20261111_v1/control/tests/test_matrix_report.py`

**Interfaces:**
- Consumes: Task 2 的训练产物(`train/<arm>/<run>/selected.pt`、summary.json);frozen 的 `python -m highway_hybrid evaluate` 入口。
- Produces:
  - `all_complete(study: Path) -> bool`:55 个 `train/*/*/summary.json` 全存在且 `status=='complete'`。
  - `audit_arm(study: Path, arm: str) -> Path`:先断言 all_complete;subprocess `[sys.executable,'-X','utf8','-u','-m','highway_hybrid','evaluate','--runs',str(study/'train'/arm),'--checkpoint','selected','--split','development','--cases','40','--output',str(study/'audit_selected'/arm)]`(cwd=examples;输出目录必须不存在);返回 `audit_selected/<arm>/summary.json` 路径。
  - `load_audit(study: Path, arm: str) -> list[dict]`:解析该 summary.json。
  - `group_rows(rows: list[dict], arm: str) -> dict[str, list[dict]]`:main 臂按 `L{level}_{method}`(run 名前部);beta 臂按 beta 值;reward 臂按 ablation 名。
  - `stats(values: list[float]) -> str`:`'0.123±0.045'`(mean±sd,3 位小数)。
  - `build_report(study: Path) -> Path`:写 `report.md` 与 `report_rows.csv`。表格列:`attack_success_rate, collision_rate, invalid_rate, grade_violation_rate, cut_count, blocking_time, utility, return`(L2/L3 用 attack_success_rate,L1 表注 blocking_time 为主指标)。**复用行**(表注"复用主对比,不重训"):B 臂 β=0 → main `L2_independent_s*`,β=0.02 → main `L2_mutual_s*`;C 臂 full → main `L2_mutual_s*`。CSV 长表列:`arm,group,seed,run,beta,ablation,risk_weight,event_weight,effect_weight,train_steps_total,split,` + 上述 8 指标。
- `matrix_audit.py` CLI:`--study [--arms main beta reward]`;`matrix_report.py` CLI:`--study`。

- [ ] **Step 1: 写失败测试**

```python
def test_stats_mean_sd():
    assert stats([.2, .4, .6]) == '0.400±0.163'   # 值按 3 位小数舍入断言

def test_group_rows_main_mapping():
    rows = [{'run':'L2_mutual_s42','overall':{...}}, {'run':'L1_independent_s43',...}]
    groups = group_rows(rows, 'main')
    assert set(groups) == {'L1_independent','L1_mutual','L2_independent','L2_mutual'}

def test_beta_report_reuses_main_rows(tmp):
    # 合成 main+beta 审计;build_report 后读 report.md
    md = (study/'report.md').read_text()
    assert 'L2_beta0' in md and '复用' in md and '0.02' in md
    # β=0 行的成功率等于 main L2_independent_s42 的值(字符串含 stats 输出)

def test_all_complete_false_on_missing(tmp):
    # 删掉一个 summary.json → all_complete(study) is False
```

- [ ] **Step 2: 跑测试确认失败**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: FAIL

- [ ] **Step 3: 实现 `matrix_audit.py` 与 `matrix_report.py`**

按 Interfaces;复用映射用固定表:`{'beta': {0.0: 'L2_independent', 0.02: 'L2_mutual'}, 'reward': {'full': 'L2_mutual'}}`,按 run 名前缀匹配。

- [ ] **Step 4: 跑测试确认通过**

Run: `..\..\..\venv\Scripts\python.exe -m unittest discover -s control/tests -v`
Expected: PASS

---

### Task 5: 冻结文档、smoke 验证与正式启动

**Files:**
- Create: `…/hybrid_matrix_20261111_v1/study.json`
- Create: `…/hybrid_matrix_20261111_v1/README.md`

- [ ] **Step 1: 写 study.json 与 README.md**

study.json:协议名、commit、日期、三臂定义(名称/数量/预算/seed)、评估口径(development 40,test 100 保留)、监控周期、验收标准(55 个 summary complete)。README.md:研究目的(研究点一证据链)、每臂回答的假设、证据声明规范(均值±SD 为主证据、负结果如实报告、test 冻结后只跑一次)、目录导览、TensorBoard 端口 6011、验收步骤。

- [ ] **Step 2: smoke——2 轮短跑验证端到端**

Run: `..\..\..\venv\Scripts\python.exe -m highway_hybrid train --config configs/main/L2_independent_s42.json --output smoke_check --max-rounds 2`(cwd=study)
Expected: exit 0;`smoke_check/summary.json` status='paused'、train_steps==512;`manifest.json` 的 source_sha256 == `eeda4f7440db917f4759fa2b3d30ef320f5a920106000c1f755745a0574c2542`(与 pilot 一致;不一致则**中止启动**排查源码差异);`latest.pt` 存在。

- [ ] **Step 3: 后台启动编排器**

Run: `..\..\..\venv\Scripts\python.exe control/matrix_runner.py --study . --jobs 8 --max-retries 2`(后台启动,stdout 追加 `logs/matrix_runner.log`)
Expected: `control/pipeline_status.json` stage='training:main';`tasklist` 出现 8 个 `-m highway_hybrid` 的 python 子进程。

- [ ] **Step 4: 首波验证(启动后 ~3 分钟)**

Run: 读 `train/main/L1_independent_s42/metrics.jsonl` 两次(间隔 ~60s)
Expected: 行数增长、train_steps 递增、无 NaN 报错;logs/ 下各 run 日志正常写入。

- [ ] **Step 5: 启动监控与 TensorBoard(后台常驻)**

Run: `..\..\..\venv\Scripts\python.exe control/matrix_monitor.py --study . --interval 600`(后台)
Run: `..\..\..\venv\Scripts\python.exe -m tensorboard.main --logdir train --host 127.0.0.1 --port 6011`(后台)
Expected: `control/monitor.jsonl` 出现首条快照;`http://127.0.0.1:6011/` 可访问。

- [ ] **Step 6: 记忆与交接**

写 memory:`hybrid-matrix-pipeline-running`(研究目录、三臂 55×500k、验收标准、test 终审与 SAC 臂留待明天)。向用户汇报:启动时间、预计完成时间(A 臂 ~7.5h,B+C 再 ~6.5h)、明早验收步骤(55 个 summary complete → `python control/matrix_audit.py --study .` → `python control/matrix_report.py --study .` → 看 `report.md`)。
