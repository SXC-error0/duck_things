"""命令桥：把「aif 认知层的离散行为」翻译成 RL 身体层的 (策略槽, 13D 命令块)。

这是分层控制里「认知层 ↔ 身体层」的接缝。认知层（LV4 的 ``SelfAwareAgent``）只输出一个
**整数行为下标**；本模块把它映射成训练好的 ONNX 策略能消费的东西：

- 13D 命令块布局（与运行时 / ``scripts/infer_policy.py`` 完全一致）：
  ``[twist(3), head_pose(4), body_pose(6)]``，即
  ``[vx, vy, vtheta, neck_pitch, head_pitch, head_yaw, head_roll,
    body_x, body_y, body_z, body_roll, body_pitch, body_yaw]``。
- 策略槽（policy slot）：``walk`` / ``sitstand`` / ``kick``。
- 训练范围（来自 ``microduck_velocity_env_cfg.py`` 与 ``infer_policy.py`` 常量）：
  线速度 ±0.4 / 侧向 ±0.3 / 角速度 ±1.0；身体姿态 ±20mm(x,y) / ±30mm(z) / ±30°。

**关键约定（踩过坑的语义，别改）**：

- ``sitstand`` 槽的 ``twist[0]`` 是**姿态旗标**（1=坐 / 0=站），**不是速度**——它超出
  线速度范围，绝不能走速度裁剪路径。见 ``sit_command``。
- 训练范围外的命令会让策略发散（OOD）；所有速度/姿态分量都必须裁剪。
- 本模块纯 numpy，不依赖 jax / mujoco，可被认知层进程独立 import。

Phase 2 会在 ``scripts/`` 里用它驱动 ``infer_policy.py`` 的 ``PolicyInference`` 跑 CPU
MuJoCo；世界系方向 → 身体系 twist 的朝向换算也留到那时（需要鸭子的实时 heading）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

# --------------------------------------------------------------------------- #
# 命令块布局与训练范围（必须与运行时一致）
# --------------------------------------------------------------------------- #
COMMAND_DIM = 13
TWIST_SLICE = slice(0, 3)   # [vx, vy, vtheta]
HEAD_SLICE = slice(3, 7)    # [neck_pitch, head_pitch, head_yaw, head_roll]
BODY_SLICE = slice(7, 13)   # [x, y, z, roll, pitch, yaw]

TWIST_RANGES = {
    "lin_vel_x": (-0.4, 0.4),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-1.0, 1.0),
}
BODY_CMD_MAX_XY = 0.02                 # ±20 mm
BODY_CMD_MAX_Z = 0.03                  # ±30 mm
BODY_CMD_MAX_ANGLE = np.radians(30.0)  # ±30°
HEAD_POSE_MAX = 1.0                    # ±1.0 rad（head_pose_range 课程最终值）

# --------------------------------------------------------------------------- #
# 行为词汇表（认知层的决策空间）
# --------------------------------------------------------------------------- #


class DuckBehavior(IntEnum):
    """鸭子可选的离散行为。语义是**身体系**运动（RL 策略原生理解的方式）。

    +x 前向 / +y 左侧 / +yaw 逆时针（标准右手系）。正负号约定待 Phase 2 用真实身体验证。
    """

    WALK_FORWARD = 0
    WALK_BACKWARD = 1
    WALK_LEFT = 2
    WALK_RIGHT = 3
    TURN_LEFT = 4
    TURN_RIGHT = 5
    STOP = 6
    SIT = 7
    STAND = 8
    KICK = 9


# --------------------------------------------------------------------------- #
# 命令产物
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DuckCommand:
    """一次身体层命令：哪个策略槽 + 13D 命令块 + 是否定长行为。"""

    slot: str                 # "walk" | "sitstand" | "kick"
    command: np.ndarray       # (13,) float32 命令块
    episodic: bool = False    # True = 定长行为（如踢球），跑完须回调认知层
    duration_s: float = 0.0   # episodic 行为的建议时长


# --------------------------------------------------------------------------- #
# 命令组装
# --------------------------------------------------------------------------- #


def _clip(v: float, lo: float, hi: float) -> float:
    return float(np.clip(v, lo, hi))


def assemble_command(
    twist: np.ndarray | list[float],
    head_pose: np.ndarray | list[float] | None = None,
    body_pose: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    """把 (twist, head_pose, body_pose) 组装成裁剪后的 13D 命令块。

    只用于**速度类**命令（walk）。``sitstand`` 的姿态旗标不走这里（见 ``sit_command``）。
    """
    twist = np.asarray(twist, dtype=np.float32).reshape(-1)
    if twist.shape != (3,):
        raise ValueError(f"twist 必须 3 维，得到 {twist.shape}")
    head = np.zeros(4, dtype=np.float32) if head_pose is None else np.asarray(head_pose, dtype=np.float32).reshape(-1)
    body = np.zeros(6, dtype=np.float32) if body_pose is None else np.asarray(body_pose, dtype=np.float32).reshape(-1)
    if head.shape != (4,) or body.shape != (6,):
        raise ValueError(f"head_pose 4 维 / body_pose 6 维，得到 {head.shape} / {body.shape}")

    vx = _clip(twist[0], *TWIST_RANGES["lin_vel_x"])
    vy = _clip(twist[1], *TWIST_RANGES["lin_vel_y"])
    vtheta = _clip(twist[2], *TWIST_RANGES["ang_vel_z"])

    head_c = np.clip(head, -HEAD_POSE_MAX, HEAD_POSE_MAX)
    body_c = body.copy()
    body_c[0:2] = np.clip(body_c[0:2], -BODY_CMD_MAX_XY, BODY_CMD_MAX_XY)
    body_c[2] = _clip(body_c[2], -BODY_CMD_MAX_Z, BODY_CMD_MAX_Z)
    body_c[3:6] = np.clip(body_c[3:6], -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE)

    return np.concatenate([[vx, vy, vtheta], head_c, body_c]).astype(np.float32)


def sit_command(sit: bool) -> DuckCommand:
    """坐 / 站：``sitstand`` 槽，``twist[0]`` 是姿态旗标（1=坐 / 0=站），**不裁剪**。"""
    cmd = np.zeros(COMMAND_DIM, dtype=np.float32)
    cmd[0] = 1.0 if sit else 0.0
    return DuckCommand(slot="sitstand", command=cmd)


def kick_command(duration_s: float = 2.0) -> DuckCommand:
    """踢球：``kick`` 槽，定长行为，训练用全零命令。"""
    return DuckCommand(
        slot="kick",
        command=np.zeros(COMMAND_DIM, dtype=np.float32),
        episodic=True,
        duration_s=duration_s,
    )


# --------------------------------------------------------------------------- #
# 行为 → 命令
# --------------------------------------------------------------------------- #

_MOVE_TWIST = {
    DuckBehavior.WALK_FORWARD: lambda s: (s, 0.0, 0.0),
    DuckBehavior.WALK_BACKWARD: lambda s: (-s, 0.0, 0.0),
    DuckBehavior.WALK_LEFT: lambda s: (0.0, s, 0.0),
    DuckBehavior.WALK_RIGHT: lambda s: (0.0, -s, 0.0),
}


def command_for(
    behavior: int | DuckBehavior,
    speed: float = 0.3,
    yaw_rate: float = 0.8,
) -> DuckCommand:
    """把认知层的整数行为下标映射成一次身体层命令。

    Parameters
    ----------
    behavior:
        ``DuckBehavior`` 或其整数值。
    speed:
        直线速度（m/s），默认 0.3（训练范围 ±0.4 内）。
    yaw_rate:
        原地转向角速度（rad/s），默认 0.8（训练范围 ±1.0 内）。
    """
    b = DuckBehavior(behavior)

    if b in _MOVE_TWIST:
        twist = _MOVE_TWIST[b](speed)
        return DuckCommand(slot="walk", command=assemble_command(twist))

    if b == DuckBehavior.TURN_LEFT:
        return DuckCommand(slot="walk", command=assemble_command((0.0, 0.0, yaw_rate)))
    if b == DuckBehavior.TURN_RIGHT:
        return DuckCommand(slot="walk", command=assemble_command((0.0, 0.0, -yaw_rate)))

    if b == DuckBehavior.STOP:
        # 零速度 → 运行时自动切到 stand 策略（infer_policy 的 _update_policy_session）。
        return DuckCommand(slot="walk", command=assemble_command((0.0, 0.0, 0.0)))

    if b == DuckBehavior.SIT:
        return sit_command(sit=True)
    if b == DuckBehavior.STAND:
        return sit_command(sit=False)
    if b == DuckBehavior.KICK:
        return kick_command()

    raise ValueError(f"未知行为 {behavior!r}")


# --------------------------------------------------------------------------- #
# 世界系方向 → 身体系 twist（导航接缝）
# --------------------------------------------------------------------------- #


def wrap_to_pi(a: float) -> float:
    """把角度折到 (-π, π]。"""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def world_move_twist(
    heading: float,
    direction_angle: float,
    speed: float = 0.3,
    yaw_rate: float = 0.8,
    turn_threshold: float = 0.3,
) -> tuple[float, float, float]:
    """把「朝世界方向 ``direction_angle`` 走」转成身体系 twist ``(vx, vy, vtheta)``。

    **转向-再走**控制器：当前朝向与目标方向夹角大于 ``turn_threshold`` 时原地转向
    （``vtheta`` 带符号，朝向目标）；已对准则直线前进（``vx=speed``）。调用方在身体层
    每个控制节拍用最新 heading 重算，即可让鸭子先转身再前进。适合**转身快**的身体。

    ``heading`` / ``direction_angle`` 都是世界系角度（弧度，+x 为 0，逆时针为正）。
    """
    rel = wrap_to_pi(direction_angle - heading)
    if abs(rel) > turn_threshold:
        return (0.0, 0.0, float(np.copysign(yaw_rate, rel)))
    return (speed, 0.0, 0.0)


def world_move_arc(
    heading: float,
    direction_angle: float,
    speed: float = 0.3,
    yaw_rate: float = 0.8,
) -> tuple[float, float, float]:
    """弧线导航：**始终边转边走，绝不纯原地转**。

    适合 velstand 这类「稳定性优先」策略：实测纯转向命令（vx=0, vyaw=±0.8）会让它
    锁死不动（heading 纹丝不动），而「前进 + 转向」的弧线才有效。所以这里给 vx 设
    保底下限（``max(0.15, cos(rel))``），背对目标时也带一点前进，靠持续转向绕一个
    大弧兜回来（前进-转弯的 Dubins 路径，双足不倒走）。调用方每节拍用最新 heading 重算。
    """
    rel = wrap_to_pi(direction_angle - heading)
    vx = max(speed * math.cos(rel), 0.15)  # 保底前进 0.15：太小（如 0.06）会让策略锁死
    vtheta = float(np.clip(4.0 * rel, -yaw_rate, yaw_rate))
    return (vx, 0.0, vtheta)
