"""闭环 + 真实 robotd 身体（Option 1：aif 认知层 ↔ robotd ↔ MuJoCo 身体）。

与 ``duck_self_drive.py`` 的 MockBody 版**同一套闭环**，只是把点质量身体换成真实
``robotd``：认知层（LV4 的 ``aif``）不再自己推物理，而是像 gamepad 一样，通过
``robotd_client`` 发 ``robot.move`` 速度意图、读 ``robot.state``（摔倒 / 里程计 / IMU）。
摔倒检测、standup、策略热插拔都由 robotd 兜底——绕开「单独 velocity 策略走不稳」的坑。

**前置**（先起真实运行时，见 microduck-main 的 ``docs/robot/simulation.md``）：

    cd microduck-main && scripts/duck-sim          # 构建并启动 duck-body + robotd + tofd
    # robotd 的 socket 在 ~/.cache/duck-sim/duck-a.sock（duck.sock 是链接）

**运行**（LV4 venv 有 jax + aif；本脚本把 microduck_rl/src 加进路径取 robotd_client）：

    cd /home/zkxt/dev/Duck
    LV4/.venv/bin/python microduck_rl/scripts/duck_self_drive_robotd.py \
        --socket ~/.cache/duck-sim/duck-a.sock --steps 40

**注意（须在真机/仿真上校对的约定，写在这里是明确标注而非实测）**：

- ``odom.position`` 是世界系（重力对齐、启动时 yaw 任意）。认知层的网格须在这个帧里
  定义 DOCK/BALL 的坐标；本脚本用 ``--dock-x/y --ball-x/y`` 传入，默认对应 duck_life 的
  0.5m 格距布局。
- ``imu.quat`` 是 trunk→世界（scalar-first），yaw = atan2(2(wz+xy), 1-2(y²+z²))。
- robotd 自己会从摔倒中站起（sitstand 策略），所以认知层只需读 ``safety.fallen`` 并等待，
  不需要自己管站起。
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

_MICRODUCK_RL_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_MICRODUCK_RL_SRC) not in sys.path:
    sys.path.insert(0, str(_MICRODUCK_RL_SRC))

# ruff: noqa: I001  —— sys.path 注入后导入跨项目依赖
import jax
import jax.numpy as jnp

from aif import SelfAwareAgent, obs_to_list
from aif.examples import duck_life as dl
from mjlab_microduck.duck_bridge import world_move_arc
from mjlab_microduck.robotd_client import RobotdClient

CELL = 0.5       # 每格边长（米）
ARRIVE_RADIUS = 0.2  # 距目标格中心多近算「到达」（真实行走有偏差，不必精确入格）


def quat_to_yaw(q: list[float]) -> float:
    """scalar-first 四元数 [w,x,y,z] → yaw（绕 z 轴）。"""
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _BodyGroundTruth:
    """连 body_server（TCP）读 ground-truth trunk 位置 + 四元数——同一 MuJoCo 世界帧，
    避开 robotd 接触里程计的漂移。"""

    def __init__(self, addr: str = "127.0.0.1:7801"):
        host, port = addr.rsplit(":", 1)
        self._sock = socket.create_connection((host, int(port)), timeout=3)
        self._sock.settimeout(3)
        self._sock.sendall(b'{"op":"hello","protocol":1}\n')
        self._read_line()

    def _read_line(self) -> dict:
        buf = b""
        while b"\n" not in buf:
            chunk = self._sock.recv(8192)
            if not chunk:
                raise ConnectionError("body server 连接关闭")
            buf += chunk
        line, _ = buf.split(b"\n", 1)
        return json.loads(line.decode())

    def read(self) -> tuple[list[float], list[float]]:
        self._sock.sendall(b'{"op":"read"}\n')
        r = self._read_line()
        trunk = r["trunk"]
        quat = r["imu"]["quat"]
        return [trunk[0], trunk[1], trunk[2]], quat

    def close(self) -> None:
        self._sock.close()


class RobotdBody:
    """真实 robotd 身体：ground-truth 位置/朝向（body_server）+ 摔倒状态（robotd 状态流）。"""

    modality_names = dl.DuckArena.modality_names

    def __init__(self, client: RobotdClient, dock_xy, ball_xy, body_addr: str = "127.0.0.1:7801",
                 seed: int = 0):
        self.client = client
        self.dock_xy = dock_xy
        self.ball_xy = ball_xy
        self.battery = dl.N_BATT - 1
        self.play = dl.N_PLAY - 1
        self._rng = np.random.default_rng(seed)
        self._gt = _BodyGroundTruth(body_addr)
        self._pos = [0.0, 0.0]
        self._yaw = 0.0
        self.fallen = False

    def refresh(self) -> None:
        trunk, quat = self._gt.read()
        self._pos = [trunk[0], trunk[1]]
        self.fallen = trunk[2] < 0.08
        self._yaw = quat_to_yaw(quat)
        # 订阅了状态流必须读，否则 robotd 的 IPC 线程写满缓冲会阻塞、不再处理 move。
        self.client.poll_state(timeout=0.0)

    def cell_of(self) -> int:
        c = max(0, min(dl.GRID - 1, round(self._pos[0] / CELL)))
        r = max(0, min(dl.GRID - 1, round(-self._pos[1] / CELL)))
        return r * dl.GRID + c

    def move_to_cell(self, target_cell: int, speed: float, yaw_rate: float, max_t: float = 60.0) -> None:
        """发 twist，直到距目标格中心 < ARRIVE_RADIUS / 超时；摔倒时等 robotd 自己站起。"""
        tx = (target_cell % dl.GRID) * CELL
        ty = -(target_cell // dl.GRID) * CELL
        t = 0.0
        while t < max_t:
            self.refresh()
            if self.fallen:
                self.client.move(0.0, 0.0, 0.0)
                time.sleep(0.5)
                t += 0.5
                continue
            dx, dy = tx - self._pos[0], ty - self._pos[1]
            if math.hypot(dx, dy) < ARRIVE_RADIUS:
                break
            direction = math.atan2(dy, dx)
            twist = world_move_arc(self._yaw, direction, speed, yaw_rate)
            self.client.move(*twist)
            time.sleep(0.02)
            t += 0.02
        self.client.move(0.0, 0.0, 0.0)

    def close(self) -> None:
        self._gt.close()

    def drain_and_recharge(self) -> None:
        pos = [self._pos[0], self._pos[1]]
        if (pos[0] - self.dock_xy[0]) ** 2 + (pos[1] - self.dock_xy[1]) ** 2 < (0.3 * CELL) ** 2:
            self.battery = dl.N_BATT - 1
        else:
            self.battery = max(self.battery - 1, 0)
        if (pos[0] - self.ball_xy[0]) ** 2 + (pos[1] - self.ball_xy[1]) ** 2 < (0.3 * CELL) ** 2:
            self.play = dl.N_PLAY - 1
        else:
            self.play = max(self.play - 1, 0)

    def observe(self) -> dict[str, jnp.ndarray]:
        cell = self.cell_of()
        if self._rng.random() < dl.POS_ACC:
            position = cell
        else:
            nbrs = dl._neighbors(cell)
            position = int(nbrs[self._rng.integers(len(nbrs))])
        flaky = int(self._rng.integers(dl.N_CELLS))
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "battery": jnp.asarray([self.battery], dtype=jnp.int32),
            "play": jnp.asarray([self.play], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }


def _target_cell(cur_cell: int, action: int) -> int:
    r, c = dl._rc(cur_cell)
    nr, nc = dl._move(r, c, action)
    return dl._cell(nr, nc)


def run(socket: str, steps: int, speed: float, yaw_rate: float, seed: int, body_addr: str):
    spec = dl._make_spec()
    model = dl._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dl._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    client = RobotdClient(socket)
    client.connect()
    client.subscribe(hz=20)
    client.enable(on=True)
    body = RobotdBody(client, dock_xy=(0.0, 0.0), ball_xy=(CELL, -CELL), body_addr=body_addr)

    # 让 agent 的初始位置先验匹配鸭子实际起点——硬先验（START）与实际观测冲突会腐蚀信念，
    # 导致 agent 误判 DOCK/BALL 的方向（Phase 0 踩过的同款坑的机器人版）。
    body.refresh()
    start_cell = body.cell_of()
    agent._pymdp.D[dl.POS_FACTOR] = jnp.asarray(
        dl._onehot(dl.N_CELLS, start_cell), dtype=jnp.float32
    )[None, :]

    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs = body.observe()
    print(f"{'t':>3s} {'act':>3s} {'cell':>3s} {'b':>1s} {'p':>1s} {'fallen':>6s} {'γ':>6s}")
    try:
        for t in range(steps):
            key, k = jax.random.split(key)
            result = agent.step(obs_to_list(obs, body.modality_names), k)
            action = int(np.asarray(result.action).reshape(-1)[0])
            body.move_to_cell(_target_cell(body.cell_of(), action), speed, yaw_rate)
            body.drain_and_recharge()
            obs = body.observe()
            gamma = result.self_state["gamma"]
            print(f"{t:3d} {dl.ACTION_NAMES[action]:>3s} {body.cell_of():3d} "
                  f"{body.battery:1d} {body.play:1d} {body.fallen!s:>6s} {gamma:6.2f}")
    finally:
        client.move(0.0, 0.0, 0.0)
        body.close()
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="aif 认知层 ↔ robotd ↔ MuJoCo 闭环")
    ap.add_argument("--socket", default="~/.cache/duck-sim/duck-a.sock")
    ap.add_argument("--body", default="127.0.0.1:7801", help="duck-body 的 host:port（读 ground-truth）")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--yaw-rate", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.socket, args.steps, args.speed, args.yaw_rate, args.seed, args.body)


if __name__ == "__main__":
    main()
