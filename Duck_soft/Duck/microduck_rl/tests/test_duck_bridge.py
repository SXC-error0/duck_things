"""命令桥（``mjlab_microduck.duck_bridge``）的单元测试。

锁定 13D 命令布局、训练范围裁剪、策略槽映射、sitstand 姿态旗标不裁剪等不变量。
"""

from __future__ import annotations

import numpy as np
import pytest

from mjlab_microduck.duck_bridge import (
    BODY_CMD_MAX_ANGLE,
    BODY_CMD_MAX_XY,
    BODY_CMD_MAX_Z,
    HEAD_POSE_MAX,
    TWIST_RANGES,
    DuckBehavior,
    assemble_command,
    command_for,
    kick_command,
    sit_command,
    world_move_arc,
    world_move_twist,
    wrap_to_pi,
)


def test_command_layout_and_shape():
    cmd = assemble_command((0.1, 0.2, 0.3), head_pose=(0.1, 0.2, 0.3, 0.4),
                           body_pose=(0.01, 0.02, 0.03, 0.1, 0.2, 0.3))
    assert cmd.shape == (13,)
    assert cmd.dtype == np.float32
    # 布局：[twist(3), head_pose(4), body_pose(6)]
    np.testing.assert_allclose(cmd[0:3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(cmd[3:7], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(cmd[7:13], [0.01, 0.02, 0.03, 0.1, 0.2, 0.3])


def test_twist_clipped_to_trained_ranges():
    # 超界输入被裁剪到训练范围，而不是原样放行（OOD 会让策略发散）。
    cmd = assemble_command((10.0, -10.0, 10.0))
    assert cmd[0] == pytest.approx(TWIST_RANGES["lin_vel_x"][1])
    assert cmd[1] == pytest.approx(TWIST_RANGES["lin_vel_y"][0])
    assert cmd[2] == pytest.approx(TWIST_RANGES["ang_vel_z"][1])
    # 未提供 head/body → 补零。
    np.testing.assert_allclose(cmd[3:13], 0.0)


def test_body_and_head_clipped():
    cmd = assemble_command(
        (0.0, 0.0, 0.0),
        head_pose=(5.0, -5.0, 0.0, 0.0),
        body_pose=(1.0, -1.0, 1.0, 2.0, -2.0, 2.0),
    )
    np.testing.assert_allclose(cmd[3:5], [HEAD_POSE_MAX, -HEAD_POSE_MAX])
    np.testing.assert_allclose(cmd[7:9], [BODY_CMD_MAX_XY, -BODY_CMD_MAX_XY])
    assert cmd[9] == pytest.approx(BODY_CMD_MAX_Z)
    np.testing.assert_allclose(cmd[10:13], [BODY_CMD_MAX_ANGLE, -BODY_CMD_MAX_ANGLE, BODY_CMD_MAX_ANGLE])


def test_sitstand_posture_flag_not_clipped():
    # twist[0] 是姿态旗标（1/0），超出速度范围，绝不能裁剪。
    sit = sit_command(sit=True)
    stand = sit_command(sit=False)
    assert sit.slot == "sitstand"
    assert sit.command[0] == 1.0
    assert stand.command[0] == 0.0
    # 其余分量全零。
    np.testing.assert_allclose(sit.command[1:13], 0.0)


def test_kick_is_episodic_zero_command():
    k = kick_command(duration_s=1.5)
    assert k.slot == "kick"
    assert k.episodic is True
    assert k.duration_s == 1.5
    np.testing.assert_allclose(k.command, 0.0)


def test_command_for_slot_mapping():
    # 运动 → walk 槽。
    for b in (DuckBehavior.WALK_FORWARD, DuckBehavior.WALK_BACKWARD,
              DuckBehavior.WALK_LEFT, DuckBehavior.WALK_RIGHT,
              DuckBehavior.TURN_LEFT, DuckBehavior.TURN_RIGHT, DuckBehavior.STOP):
        assert command_for(b).slot == "walk"
    # 坐/站 → sitstand；踢 → kick。
    assert command_for(DuckBehavior.SIT).slot == "sitstand"
    assert command_for(DuckBehavior.STAND).slot == "sitstand"
    assert command_for(DuckBehavior.KICK).slot == "kick"


def test_move_speed_within_range():
    # 默认速度/角速度落在训练范围内。
    fwd = command_for(DuckBehavior.WALK_FORWARD)
    assert 0.0 < fwd.command[0] <= TWIST_RANGES["lin_vel_x"][1]
    turn = command_for(DuckBehavior.TURN_LEFT)
    assert 0.0 < turn.command[2] <= TWIST_RANGES["ang_vel_z"][1]


def test_command_for_accepts_int():
    # 认知层输出整数下标；桥要能直接消费 int。
    assert command_for(int(DuckBehavior.SIT)).slot == "sitstand"
    with pytest.raises(ValueError):
        command_for(999)


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == 0.0
    assert wrap_to_pi(np.pi) == pytest.approx(-np.pi)  # π 折到 (-π, π] 的左端点
    assert wrap_to_pi(3.0 * np.pi / 2.0) == pytest.approx(-np.pi / 2.0)
    assert wrap_to_pi(-3.0 * np.pi / 2.0) == pytest.approx(np.pi / 2.0)


def test_world_move_twist_turns_then_walks():
    # 目标方向 = 东(0)。当前朝北(π/2)：夹角大 → 原地右转（负 vtheta）。
    vx, vy, vtheta = world_move_twist(heading=np.pi / 2, direction_angle=0.0)
    assert (vx, vy) == (0.0, 0.0)
    assert vtheta < 0.0

    # 已朝东(0)：夹角小 → 直线前进。
    vx, vy, vtheta = world_move_twist(heading=0.0, direction_angle=0.0)
    assert vx > 0.0
    assert (vy, vtheta) == (0.0, 0.0)


def test_world_move_twist_wraps_heading():
    # 目标 = 西(π)，当前朝东(0)，但经 ±π 折后夹角应接近 π → 转向。
    vx, vy, vtheta = world_move_twist(heading=0.0, direction_angle=np.pi)
    assert (vx, vy) == (0.0, 0.0)
    assert vtheta != 0.0


def test_world_move_arc_always_advances():
    # 背对目标（夹角 > 90°）也带保底前进（绝不纯原地转——velstand 会锁死）。
    vx, vy, vtheta = world_move_arc(heading=0.0, direction_angle=np.pi)
    assert vx > 0.0
    assert vy == 0.0
    assert vtheta != 0.0


def test_world_move_arc_advances_while_turning():
    # 面向目标一侧（夹角 < 90°）→ 边转边走，vx > 0。
    vx, vy, vtheta = world_move_arc(heading=0.0, direction_angle=np.pi / 4)
    assert vx > 0.0
    assert vy == 0.0
    assert vtheta > 0.0
    # 已对准：几乎全速前进，几乎不转。
    vx, vy, vtheta = world_move_arc(heading=0.0, direction_angle=0.0)
    assert vx > 0.0
    assert abs(vtheta) < 1e-6
