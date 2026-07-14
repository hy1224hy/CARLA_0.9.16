"""
对抗行为归因工具（adversarial attribution metrics）
====================================================
把"蓝车到底对红车造成了什么影响"从几何位置里剥离出来，做成 **CARLA 解耦** 的纯函数
+ 一个 env 适配器，便于单元测试与在 reward / TB 两处复用。

设计文档：docs/adversarial_metrics_reward_design_and_plan_20260705.md §1.3 Layer C、§2.3、
          docs/implementation_plan_adversarial_metrics_20260705.md

核心概念
--------
- AgentKinematics    : 单车某一步的位姿 + 前向速度 + 转向，纯数据。
- extract_kinematics : 唯一触碰 CARLA env 的适配器，把 env.vehicles 转成 AgentKinematics。
- relative_long_lat  : 目标车在观察车坐标系下的 (纵向, 横向)。
- in_threat_window   : 蓝车是否在红车前方威胁窗口内（与 is_any_red_under_threat 同口径）。
- SlowdownDetector   : 滚动窗口检测红车"归因减速"（2s 内前向速度降 ≥ Δv）。
- SteerResponseDetector : 检测红车"归因避让"（steer 突变）。
- classify_collision : 碰撞三分法 kamikaze / induced / collateral。
- birth_protection_active : 出生保护期（前 N 秒的归因事件不计分，防出生点封锁 hack）。
- red_is_responding  : 门控开关 = 减速 或 steer 突增。

所有阈值都是参数，默认值与 §2.3 表格 / compute_blue_threat_scores 对齐。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple


@dataclass
class AgentKinematics:
    """单车某一步的运动学快照（世界坐标 + 车体前向速度 + 转向）。"""
    loc: Tuple[float, float]          # (x, y) 世界坐标
    fwd: Tuple[float, float]          # 车头朝向单位向量
    right: Tuple[float, float]        # 车体右向单位向量
    fwd_speed: float                  # 沿车头方向的速度分量 (m/s)，可为负（倒车）
    speed: float = 0.0                # 速度模长 (m/s)
    steer: float = 0.0                # 当前转向输入 [-1, 1]，无控制信息时为 0
    alive: bool = True                # 车辆是否存活
    reached: bool = False             # 是否已到达目标


def extract_kinematics(env, ids: List[int],
                       control_dict: Optional[Dict[int, "object"]] = None
                       ) -> Dict[int, AgentKinematics]:
    """【唯一触碰 CARLA 的适配器】把 env.vehicles[id] 转成 AgentKinematics。

    参数:
      env          : CarlaMultiAgentEnv（用到 .vehicles / .reached_target）
      ids          : 需要提取的 agent id 列表
      control_dict : 可选 {id: carla.VehicleControl}，用于填 steer；缺省 steer=0

    只对存活且下标合法的车返回条目；不存在/已 despawn 的车直接跳过。
    """
    out: Dict[int, AgentKinematics] = {}
    for aid in ids:
        if aid < 0 or aid >= len(env.vehicles):
            continue
        v = env.vehicles[aid]
        if v is None or not v.is_alive:
            continue
        tr = v.get_transform()
        vel = v.get_velocity()
        yaw_rad = math.radians(tr.rotation.yaw)
        fwd = (math.cos(yaw_rad), math.sin(yaw_rad))
        right = (math.sin(yaw_rad), -math.cos(yaw_rad))
        fwd_speed = vel.x * fwd[0] + vel.y * fwd[1]
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2)
        steer = 0.0
        if control_dict is not None and aid in control_dict and control_dict[aid] is not None:
            steer = float(getattr(control_dict[aid], "steer", 0.0))
        out[aid] = AgentKinematics(
            loc=(tr.location.x, tr.location.y),
            fwd=fwd, right=right,
            fwd_speed=fwd_speed, speed=speed, steer=steer,
            alive=True,
            reached=bool(env.reached_target.get(aid, False)),
        )
    return out


def relative_long_lat(observer: AgentKinematics,
                      target: AgentKinematics) -> Tuple[float, float]:
    """target 在 observer 坐标系下的 (纵向, 横向)。

    纵向 > 0 ⇒ target 在 observer 前方；横向 = 沿 observer 右向的偏移。
    """
    rel = (target.loc[0] - observer.loc[0], target.loc[1] - observer.loc[1])
    longitudinal = rel[0] * observer.fwd[0] + rel[1] * observer.fwd[1]
    lateral = rel[0] * observer.right[0] + rel[1] * observer.right[1]
    return longitudinal, lateral


def in_threat_window(red: AgentKinematics, blue: AgentKinematics,
                     threat_radius: float = 15.0,
                     threat_lateral: float = 2.5,
                     near: float = 0.5) -> bool:
    """蓝车是否在红车前方威胁窗口内（与 is_any_red_under_threat 同口径）。

    条件：near ≤ 纵向 ≤ threat_radius 且 |横向| ≤ threat_lateral（红坐标系，blue 相对 red）。
    """
    longitudinal, lateral = relative_long_lat(red, blue)
    return (near <= longitudinal <= threat_radius) and (abs(lateral) <= threat_lateral)


class SlowdownDetector:
    """检测红车"归因减速"：滚动窗口内前向速度下降 ≥ drop_thresh (m/s)。

    每辆红车维护一个定长速度窗（长度 = window_secs / timestep）。每步 push 一次，
    is_slowing(rid) 判断"窗口最早值 − 当前值 ≥ drop_thresh"。用于：
      - r_threat_gated 的门控（红车正在被压慢才给稠密威胁分）
      - r_impact 的"归因减速"事件（配合 in_threat_window 与出生保护期）
    """

    def __init__(self, window_secs: float = 2.0, timestep: float = 0.1,
                 drop_thresh: float = 2.0):
        self.maxlen = max(2, int(round(window_secs / max(timestep, 1e-3))))
        self.drop_thresh = float(drop_thresh)
        self._hist: Dict[int, Deque[float]] = {}

    def update(self, kins: Dict[int, AgentKinematics]) -> None:
        for rid, k in kins.items():
            self._hist.setdefault(rid, deque(maxlen=self.maxlen)).append(k.fwd_speed)

    def drop(self, rid: int) -> float:
        """窗口内的速度降幅（最早值 − 当前值），窗口未满或无数据返回 0。"""
        h = self._hist.get(rid)
        if h is None or len(h) < self.maxlen:
            return 0.0
        return float(h[0] - h[-1])

    def is_slowing(self, rid: int) -> bool:
        return self.drop(rid) >= self.drop_thresh

    def reset(self) -> None:
        self._hist.clear()


class SteerResponseDetector:
    """检测红车"归因避让"：|steer| 相对滚动均值突增 ≥ jump_thresh。

    捕捉红车被逼到打方向躲避的瞬间。窗口很短（默认 0.5s），只看突变而非绝对值，
    因为直行道上正常驾驶 steer≈0，突增即异常反应。
    """

    def __init__(self, window_secs: float = 0.5, timestep: float = 0.1,
                 jump_thresh: float = 0.15):
        self.maxlen = max(2, int(round(window_secs / max(timestep, 1e-3))))
        self.jump_thresh = float(jump_thresh)
        self._hist: Dict[int, Deque[float]] = {}

    def update(self, kins: Dict[int, AgentKinematics]) -> None:
        for rid, k in kins.items():
            self._hist.setdefault(rid, deque(maxlen=self.maxlen)).append(abs(k.steer))

    def is_swerving(self, rid: int) -> bool:
        h = self._hist.get(rid)
        if h is None or len(h) < 2:
            return False
        baseline = sum(list(h)[:-1]) / max(len(h) - 1, 1)
        return (h[-1] - baseline) >= self.jump_thresh

    def reset(self) -> None:
        self._hist.clear()


def birth_protection_active(step: int, timestep: float = 0.1,
                            protect_secs: float = 3.0) -> bool:
    """出生保护期：回合前 protect_secs 秒内为 True。

    此期间触发的归因事件不计 r_impact，防"蓝车卡在红车出生点"刷分（hack H4）。
    step 从 0 或 1 起算均可（用 < 而非 <=，边界差 1 步无碍）。
    """
    return step < int(round(protect_secs / max(timestep, 1e-3)))


def red_is_responding(rid: int,
                      slowdown: SlowdownDetector,
                      steer: Optional[SteerResponseDetector] = None) -> bool:
    """门控开关：红车是否正在对蓝车做出反应（减速 或 打方向躲避）。

    r_threat_gated = threat_score × 1[red_is_responding]，把"占位刷分"堵死（hack H2）。
    """
    if slowdown.is_slowing(rid):
        return True
    if steer is not None and steer.is_swerving(rid):
        return True
    return False


# 碰撞归因三分类结果常量
COLLISION_KAMIKAZE = "kamikaze"     # 蓝车主动怼上去（从后方/侧向冲撞）→ 0 奖励
COLLISION_INDUCED = "induced"       # 红车撞上前方的蓝车（蓝车挡道逼撞）→ 有效
COLLISION_COLLATERAL = "collateral" # 其它有效碰撞（如蓝车侧翼被红车追尾等）→ 有效


def classify_collision(blue: AgentKinematics, red: AgentKinematics,
                       closing_eps: float = 0.5) -> str:
    """蓝-红碰撞三分法：判断谁是"制造碰撞"的一方。

    几何 + 闭合速度启发式（红坐标系下 blue 相对 red 的纵向位置 + 双方前向速度）：

      induced   : 蓝车在红车【前方】(纵向>0) 且红车正前向行驶(fwd_speed>eps)
                  → 红车主动撞上挡在前面的蓝车 = 蓝车挡道逼撞成功（有效对抗）。
      kamikaze  : 蓝车在红车【后方/侧后】(纵向≤0) 且蓝车正前向冲(fwd_speed>eps)
                  → 蓝车从后面怼上去 = 自杀式撞击（hack H1，0 奖励，c_self 照罚）。
      collateral: 其它情况（双方都近静止、或红车静止蓝车前方等）→ 有效但非主动诱导。

    注意：本函数只做"方向/责任"判断，不判断是否真的发生碰撞——调用方需先确认碰撞事件。
    """
    longitudinal, _ = relative_long_lat(red, blue)   # blue 在 red 坐标系的纵向
    if longitudinal > 0.0:
        # 蓝车在红车前方
        if red.fwd_speed > closing_eps:
            return COLLISION_INDUCED     # 红车前向撞上前方蓝车
        return COLLISION_COLLATERAL
    else:
        # 蓝车在红车后方/侧后
        if blue.fwd_speed > closing_eps:
            return COLLISION_KAMIKAZE    # 蓝车从后方主动冲撞
        return COLLISION_COLLATERAL



