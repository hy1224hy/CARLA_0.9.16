# 对抗归因度量工具 (adversarial_metrics) 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `utils/adversarial_metrics.py`——对抗归因度量的单一真相源，供训练奖励（`training/adversarial_reward.py`）与评估（`evaluation/evaluate_adversarial_c012.py`）共用同一套口径。

**Architecture:** 纯函数 + 少量有状态检测器，全部操作**与 CARLA 解耦的 `AgentKinematics` 数据类**（位置/朝向/速度/转向），仅一个薄适配器 `extract_kinematics(env, id)` 触碰 CARLA。因此核心逻辑可用普通对象做 TDD，无需启动模拟器。

**Tech Stack:** Python 3.10、dataclasses、math、pytest。运行环境为仓库根 pipenv。

**范围**：本计划只交付归因**原语模块 + 单测**（设计文档 §0.2 / Phase 0.2）。以下为**后续独立计划**，均依赖本模块先落地：
- `training/adversarial_reward.py`（§2.3 奖励组装，import 本模块）
- `evaluation/evaluate_adversarial_c012.py`（C0/C1/C2 三条件评估，import 本模块）
- 三处训练脚本改 import 共享奖励模块（§2.5）

**参考**：设计文档 `docs/adversarial_metrics_reward_design_and_plan_20260705.md` §1.3（指标定义）、§3（反 hacking）。

**关键既有接口**（已核对，勿改）：
- `env.vehicles[id]`：`.is_alive`、`.get_transform()`→`.location.x/.y`+`.rotation.yaw`(度)、`.get_velocity()`→`.x/.y`、`.get_control()`→`.steer`
- `env.reached_target: Dict[int,bool]`、`env.collision_events: Dict[int,bool]`
- `env._get_target_location(id)` → `carla.Location`（含 `.x/.y`）
- 威胁窗口口径须与 `train_multi_adversarial_kl_risk.compute_blue_threat_scores` 一致：red 坐标系下 blue 纵向 `∈[near, radius]`、`|横向|≤lateral`

---

## 前置：运行约定

所有命令从 `mappo_training/` 目录、经仓库根 pipenv 运行：

```bash
export PIPENV_PIPFILE=/d/study/yan2/CARLA_0.9.16/Pipfile
export PIPENV_VERBOSITY=-1
cd /d/study/yan2/CARLA_0.9.16/PythonAPI/examples/mappo_training
```

测试统一命令前缀：`pipenv run python -m pytest`。本模块单测**不需要 CARLA 服务器**。

---

## File Structure

- Create: `utils/adversarial_metrics.py` — 归因原语（数据类 + 几何/运动学函数 + 有状态检测器 + env 适配器）
- Create: `tests/__init__.py` — 空文件，使 tests 成为包
- Create: `tests/test_adversarial_metrics.py` — 全部单测

---

### Task 1: 模块骨架 + AgentKinematics + 威胁窗口判定

**Files:**
- Create: `utils/adversarial_metrics.py`
- Create: `tests/__init__.py`
- Create: `tests/test_adversarial_metrics.py`

- [ ] **Step 1: 写失败测试**

创建 `tests/__init__.py`（空文件）。创建 `tests/test_adversarial_metrics.py`：

```python
import math
from utils.adversarial_metrics import AgentKinematics, in_threat_window


def _red_facing_east():
    # 朝 +x 方向（yaw=0），静止
    return AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)


def test_blue_directly_ahead_is_in_window():
    red = _red_facing_east()
    blue = AgentKinematics(x=8.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert in_threat_window(red, blue, radius=15.0, lateral=2.5, near=0.5) is True


def test_blue_behind_is_not_in_window():
    red = _red_facing_east()
    blue = AgentKinematics(x=-8.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert in_threat_window(red, blue) is False


def test_blue_too_far_lateral_is_not_in_window():
    red = _red_facing_east()
    blue = AgentKinematics(x=8.0, y=5.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert in_threat_window(red, blue, lateral=2.5) is False


def test_blue_too_close_below_near_is_not_in_window():
    red = _red_facing_east()
    blue = AgentKinematics(x=0.2, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert in_threat_window(red, blue, near=0.5) is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「ModuleNotFoundError / cannot import name」

- [ ] **Step 3: 写最小实现**

创建 `utils/adversarial_metrics.py`：

```python
"""对抗归因度量的单一真相源。

所有几何/运动学函数操作与 CARLA 解耦的 AgentKinematics，便于单测。
仅 extract_kinematics() 触碰 env（见文末）。威胁窗口口径与
train_multi_adversarial_kl_risk.compute_blue_threat_scores 对齐。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class AgentKinematics:
    x: float
    y: float
    yaw_rad: float          # 弧度
    vx: float
    vy: float
    steer: float = 0.0      # 当前施加的方向盘 [-1,1]
    is_alive: bool = True
    reached: bool = False


def _rel_lon_lat(ref: AgentKinematics, other: AgentKinematics):
    """other 在 ref 车体坐标系下的 (纵向, 横向)。纵向>0=前方。"""
    fwd = (math.cos(ref.yaw_rad), math.sin(ref.yaw_rad))
    right = (math.sin(ref.yaw_rad), -math.cos(ref.yaw_rad))
    dx = other.x - ref.x
    dy = other.y - ref.y
    lon = dx * fwd[0] + dy * fwd[1]
    lat = dx * right[0] + dy * right[1]
    return lon, lat


def in_threat_window(red: AgentKinematics, blue: AgentKinematics,
                     radius: float = 15.0, lateral: float = 2.5,
                     near: float = 0.5) -> bool:
    """blue 是否处于 red 正前方的威胁窗口内。"""
    lon, lat = _rel_lon_lat(red, blue)
    return (near <= lon <= radius) and (abs(lat) <= lateral)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
git add utils/adversarial_metrics.py tests/__init__.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): AgentKinematics + threat window predicate"
```

---

### Task 2: any_red_under_threat + 朝目标闭合速度

**Files:**
- Modify: `utils/adversarial_metrics.py`
- Test: `tests/test_adversarial_metrics.py`

- [ ] **Step 1: 写失败测试**（追加到测试文件末尾）

```python
from utils.adversarial_metrics import any_red_under_threat, closing_speed_to_goal


def test_any_red_under_threat_true_when_one_red_threatened():
    red_ok = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    red_free = AgentKinematics(x=50.0, y=50.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    blue = AgentKinematics(x=8.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert any_red_under_threat([red_ok, red_free], [blue]) is True


def test_any_red_under_threat_skips_reached_red():
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0, reached=True)
    blue = AgentKinematics(x=8.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert any_red_under_threat([red], [blue]) is False


def test_any_red_under_threat_skips_dead_blue():
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    blue = AgentKinematics(x=8.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0, is_alive=False)
    assert any_red_under_threat([red], [blue]) is False


def test_closing_speed_positive_when_moving_toward_goal():
    # red 在原点，goal 在 +x，速度朝 +x → 闭合速度 = +5
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=5.0, vy=0.0)
    assert closing_speed_to_goal(red, goal_x=100.0, goal_y=0.0) == 5.0


def test_closing_speed_negative_when_moving_away():
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=-3.0, vy=0.0)
    assert closing_speed_to_goal(red, goal_x=100.0, goal_y=0.0) == -3.0


def test_closing_speed_zero_at_goal():
    red = AgentKinematics(x=100.0, y=0.0, yaw_rad=0.0, vx=5.0, vy=0.0)
    assert closing_speed_to_goal(red, goal_x=100.0, goal_y=0.0) == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「cannot import name 'any_red_under_threat'」

- [ ] **Step 3: 写最小实现**（追加到 `utils/adversarial_metrics.py`）

```python
def any_red_under_threat(reds: List[AgentKinematics], blues: List[AgentKinematics],
                         radius: float = 15.0, lateral: float = 2.5,
                         near: float = 0.5) -> bool:
    """是否有任一存活未到点的 red 处于任一存活 blue 的威胁窗口内。"""
    for red in reds:
        if not red.is_alive or red.reached:
            continue
        for blue in blues:
            if not blue.is_alive:
                continue
            if in_threat_window(red, blue, radius, lateral, near):
                return True
    return False


def closing_speed_to_goal(agent: AgentKinematics, goal_x: float, goal_y: float) -> float:
    """agent 速度在「指向 goal 的单位向量」上的投影。正=朝目标靠近。"""
    dx = goal_x - agent.x
    dy = goal_y - agent.y
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return 0.0
    return agent.vx * (dx / dist) + agent.vy * (dy / dist)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（10 passed）

- [ ] **Step 5: 提交**

```bash
git add utils/adversarial_metrics.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): any_red_under_threat + closing_speed_to_goal"
```

---

### Task 3: 碰撞几何三分类（几何代理）

**Files:**
- Modify: `utils/adversarial_metrics.py`
- Test: `tests/test_adversarial_metrics.py`

碰撞瞬间用双方相对位置+朝向做代理判定（设计文档 §1.3 碰撞归因三分法）：
- **kamikaze**（蓝主动撞）：red 在 blue 正前方 → 是 blue 冲上去。
- **induced**（诱导，红车头撞蓝车）：blue 在 red 正前方 → 是 red 撞上去。
- **collateral**（连带）：两者都不在对方正前方（侧向/其它）。

- [ ] **Step 1: 写失败测试**（追加）

```python
from utils.adversarial_metrics import classify_collision


def test_classify_kamikaze_blue_rams_red():
    # red 静止朝 +x；blue 在 red 后方、朝 +x（red 在 blue 前方）→ blue 撞上来
    red = AgentKinematics(x=10.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    blue = AgentKinematics(x=8.0, y=0.0, yaw_rad=0.0, vx=5.0, vy=0.0)
    assert classify_collision(red, blue) == "kamikaze"


def test_classify_induced_red_hits_blue_ahead():
    # blue 在 red 正前方（red 朝 +x，blue 在 +x）→ red 撞上去
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=5.0, vy=0.0)
    blue = AgentKinematics(x=2.0, y=0.0, yaw_rad=0.0, vx=0.0, vy=0.0)
    assert classify_collision(red, blue) == "induced"


def test_classify_collateral_side_by_side():
    # 并排（blue 在 red 正侧方），双方都不在对方正前方
    red = AgentKinematics(x=0.0, y=0.0, yaw_rad=0.0, vx=1.0, vy=0.0)
    blue = AgentKinematics(x=0.0, y=2.0, yaw_rad=0.0, vx=1.0, vy=0.0)
    assert classify_collision(red, blue) == "collateral"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「cannot import name 'classify_collision'」

- [ ] **Step 3: 写最小实现**（追加）

```python
def classify_collision(red: AgentKinematics, blue: AgentKinematics,
                       frontal_lateral: float = 1.5) -> str:
    """碰撞瞬间的几何三分类：kamikaze / induced / collateral。

    frontal_lateral: 判定「正前方」的横向容差（半车宽量级）。
    优先级：先看 red 是否在 blue 正前方（blue 撞上去=kamikaze），
    再看 blue 是否在 red 正前方（red 撞上去=induced），否则 collateral。
    """
    # red 在 blue 车体系下的位置
    lon_r_in_b, lat_r_in_b = _rel_lon_lat(blue, red)
    if lon_r_in_b > 0.0 and abs(lat_r_in_b) <= frontal_lateral:
        return "kamikaze"
    # blue 在 red 车体系下的位置
    lon_b_in_r, lat_b_in_r = _rel_lon_lat(red, blue)
    if lon_b_in_r > 0.0 and abs(lat_b_in_r) <= frontal_lateral:
        return "induced"
    return "collateral"
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（13 passed）

- [ ] **Step 5: 提交**

```bash
git add utils/adversarial_metrics.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): geometric collision classifier"
```

---

### Task 4: 归因减速检测器（有状态）

**Files:**
- Modify: `utils/adversarial_metrics.py`
- Test: `tests/test_adversarial_metrics.py`

设计文档 §1.3：红车朝目标闭合速度在 2s（=20 步 @0.1s）内下降 ≥2 m/s，**且**当时 blue 在其威胁窗口内 → 触发一次归因减速事件。触发后进入冷却，避免同一次减速重复计数。

- [ ] **Step 1: 写失败测试**（追加）

```python
from utils.adversarial_metrics import SlowdownDetector


def test_slowdown_fires_on_drop_with_blue_in_window():
    det = SlowdownDetector(drop_threshold=2.0, window_steps=20, cooldown_steps=20)
    fired = []
    # 前 5 步匀速 6 m/s，blue 在窗口内
    for _ in range(5):
        fired.append(det.update(closing_speed=6.0, blue_in_window=True))
    # 第 6 步骤降到 3 m/s（降 3 ≥ 2），blue 仍在窗口 → 触发
    fired.append(det.update(closing_speed=3.0, blue_in_window=True))
    assert fired[-1] is True
    assert sum(fired) == 1  # 只触发一次


def test_slowdown_not_attributed_when_blue_absent():
    det = SlowdownDetector(drop_threshold=2.0, window_steps=20, cooldown_steps=20)
    for _ in range(5):
        det.update(closing_speed=6.0, blue_in_window=False)
    fired = det.update(closing_speed=2.0, blue_in_window=False)
    assert fired is False


def test_slowdown_respects_cooldown():
    det = SlowdownDetector(drop_threshold=2.0, window_steps=20, cooldown_steps=20)
    for _ in range(5):
        det.update(closing_speed=6.0, blue_in_window=True)
    first = det.update(closing_speed=3.0, blue_in_window=True)   # 触发
    second = det.update(closing_speed=1.0, blue_in_window=True)  # 冷却中，不再触发
    assert first is True
    assert second is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「cannot import name 'SlowdownDetector'」

- [ ] **Step 3: 写最小实现**（追加；文件顶部 import 增加 `from collections import deque`）

在文件顶部 import 区加入：

```python
from collections import deque
```

追加类：

```python
class SlowdownDetector:
    """逐步喂入红车闭合速度，检测「近窗口内峰值 − 当前 ≥ 阈值」的归因减速。

    每次触发后冷却 cooldown_steps 步，避免一次减速被重复计数。
    在每个 episode 开始处 new 一个实例（或调用 reset）。
    """

    def __init__(self, drop_threshold: float = 2.0, window_steps: int = 20,
                 cooldown_steps: int = 20):
        self.drop_threshold = drop_threshold
        self.window_steps = window_steps
        self.cooldown_steps = cooldown_steps
        self._hist: deque = deque(maxlen=window_steps)
        self._cooldown = 0

    def reset(self):
        self._hist.clear()
        self._cooldown = 0

    def update(self, closing_speed: float, blue_in_window: bool) -> bool:
        fired = False
        if self._cooldown > 0:
            self._cooldown -= 1
        elif blue_in_window and len(self._hist) > 0:
            peak = max(self._hist)
            if peak - closing_speed >= self.drop_threshold:
                fired = True
                self._cooldown = self.cooldown_steps
        self._hist.append(closing_speed)
        return fired
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（16 passed）

- [ ] **Step 5: 提交**

```bash
git add utils/adversarial_metrics.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): attributed slowdown detector"
```

---

### Task 5: 归因避让检测器 + 出生保护期

**Files:**
- Modify: `utils/adversarial_metrics.py`
- Test: `tests/test_adversarial_metrics.py`

§1.3：红车方向盘突变（`|steer_t − steer_{t-1}| > 阈值`）且 blue 在窗口内 → 归因避让；带冷却。`birth_protection_active`：回合前 N 步的事件不计入 r_impact（防 H4 出生封锁刷分）。

- [ ] **Step 1: 写失败测试**（追加）

```python
from utils.adversarial_metrics import SteerResponseDetector, birth_protection_active


def test_steer_response_fires_on_spike_with_blue():
    det = SteerResponseDetector(delta_threshold=0.3, cooldown_steps=10)
    assert det.update(steer=0.0, blue_in_window=True) is False   # 首步建立基线
    assert det.update(steer=0.05, blue_in_window=True) is False  # 小变化
    assert det.update(steer=0.5, blue_in_window=True) is True    # 突变 0.45 > 0.3


def test_steer_response_not_attributed_without_blue():
    det = SteerResponseDetector(delta_threshold=0.3, cooldown_steps=10)
    det.update(steer=0.0, blue_in_window=False)
    assert det.update(steer=0.9, blue_in_window=False) is False


def test_birth_protection_boundary():
    assert birth_protection_active(step=0, protect_steps=30) is True
    assert birth_protection_active(step=29, protect_steps=30) is True
    assert birth_protection_active(step=30, protect_steps=30) is False
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「cannot import name 'SteerResponseDetector'」

- [ ] **Step 3: 写最小实现**（追加）

```python
class SteerResponseDetector:
    """检测红车方向盘相对上一步的突变（blue 在窗口内时归因为避让）。"""

    def __init__(self, delta_threshold: float = 0.3, cooldown_steps: int = 10):
        self.delta_threshold = delta_threshold
        self.cooldown_steps = cooldown_steps
        self._prev: Optional[float] = None
        self._cooldown = 0

    def reset(self):
        self._prev = None
        self._cooldown = 0

    def update(self, steer: float, blue_in_window: bool) -> bool:
        fired = False
        if self._cooldown > 0:
            self._cooldown -= 1
        elif blue_in_window and self._prev is not None:
            if abs(steer - self._prev) > self.delta_threshold:
                fired = True
                self._cooldown = self.cooldown_steps
        self._prev = steer
        return fired


def birth_protection_active(step: int, protect_steps: int = 30) -> bool:
    """回合前 protect_steps 步内返回 True（这些步的归因事件不计 r_impact）。"""
    return step < protect_steps
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（19 passed）

- [ ] **Step 5: 提交**

```bash
git add utils/adversarial_metrics.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): steer-response detector + birth protection"
```

---

### Task 6: env 适配器 extract_kinematics

**Files:**
- Modify: `utils/adversarial_metrics.py`
- Test: `tests/test_adversarial_metrics.py`

唯一触碰 CARLA 的函数。用轻量 fake 对象测试（不启动模拟器），验证字段映射与角度换算（度→弧度）正确。

- [ ] **Step 1: 写失败测试**（追加）

```python
from utils.adversarial_metrics import extract_kinematics


class _FakeLoc:
    def __init__(self, x, y):
        self.x = x; self.y = y


class _FakeRot:
    def __init__(self, yaw):
        self.yaw = yaw


class _FakeTransform:
    def __init__(self, x, y, yaw):
        self.location = _FakeLoc(x, y); self.rotation = _FakeRot(yaw)


class _FakeVel:
    def __init__(self, x, y):
        self.x = x; self.y = y


class _FakeControl:
    def __init__(self, steer):
        self.steer = steer


class _FakeVehicle:
    def __init__(self, x, y, yaw, vx, vy, steer, alive=True):
        self._t = _FakeTransform(x, y, yaw)
        self._v = _FakeVel(vx, vy)
        self._c = _FakeControl(steer)
        self.is_alive = alive

    def get_transform(self):
        return self._t

    def get_velocity(self):
        return self._v

    def get_control(self):
        return self._c


class _FakeEnv:
    def __init__(self, vehicle, reached=False):
        self.vehicles = [vehicle]
        self.reached_target = {0: reached}


def test_extract_kinematics_maps_fields_and_converts_yaw():
    veh = _FakeVehicle(x=1.0, y=2.0, yaw=90.0, vx=3.0, vy=4.0, steer=0.25)
    env = _FakeEnv(veh, reached=True)
    k = extract_kinematics(env, 0)
    assert k.x == 1.0 and k.y == 2.0
    assert k.vx == 3.0 and k.vy == 4.0
    assert k.steer == 0.25
    assert k.is_alive is True
    assert k.reached is True
    assert abs(k.yaw_rad - math.pi / 2) < 1e-9   # 90° → π/2
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: FAIL —「cannot import name 'extract_kinematics'」

- [ ] **Step 3: 写最小实现**（追加）

```python
def extract_kinematics(env, agent_id: int) -> AgentKinematics:
    """从 env 抽取单车运动学。唯一触碰 CARLA 的函数。"""
    v = env.vehicles[agent_id]
    tr = v.get_transform()
    vel = v.get_velocity()
    ctrl = v.get_control()
    return AgentKinematics(
        x=tr.location.x,
        y=tr.location.y,
        yaw_rad=math.radians(tr.rotation.yaw),
        vx=vel.x,
        vy=vel.y,
        steer=ctrl.steer,
        is_alive=v.is_alive,
        reached=env.reached_target.get(agent_id, False),
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（20 passed）

- [ ] **Step 5: 全量回归 + 提交**

Run: `pipenv run python -m pytest tests/test_adversarial_metrics.py -v`
Expected: PASS（20 passed，无警告失败）

```bash
git add utils/adversarial_metrics.py tests/test_adversarial_metrics.py
git commit -m "feat(metrics): env adapter extract_kinematics"
```

---

## 验收标准

- `pipenv run python -m pytest tests/test_adversarial_metrics.py -v` → 20 passed。
- `utils/adversarial_metrics.py` 导出：`AgentKinematics`、`in_threat_window`、`any_red_under_threat`、`closing_speed_to_goal`、`classify_collision`、`SlowdownDetector`、`SteerResponseDetector`、`birth_protection_active`、`extract_kinematics`。
- 威胁窗口口径（near/radius/lateral，red 车体系）与 `compute_blue_threat_scores` 一致。
- 无任何 CARLA import 于纯函数路径（仅 `extract_kinematics` 通过鸭子类型访问 env，测试用 fake 覆盖）。

## 后续计划（依赖本模块）

1. `training/adversarial_reward.py` — §2.3 奖励组装：r_impact（用 `SlowdownDetector`/`SteerResponseDetector`/`classify_collision` + 每回合封顶）+ 门控 threat（`compute_blue_threat_scores × 红车响应`）+ base_safe + λ_t·c_self。**产出后**三处训练脚本改 import 它（§2.5）。
2. `evaluation/evaluate_adversarial_c012.py` — C0/C1/C2 三条件评估，用本模块统计 §1.3 四层指标 + 配对 CI。
3. hack 监测仪表盘 TB 曲线（§3.3）接入训练脚本。
