"""宠物鸭子闭环：自己溜达 + 累了休息（aif 认知层 ↔ robotd ↔ MuJoCo 3D 身体）。

与 ``duck_self_drive_robotd.py`` 同一套架构，但认知层换成 ``aif.examples.duck_pet``：
两个互斥的内部稳态——精力(energy)在 HOME 回满、兴致(fun)在探索点回满——驱动鸭子
「没事出去溜达、累了回家休息」。决策经 robotd_client 发给真实 robotd，由 velstand
策略在 MuJoCo 里走出真实动作，你在 3D viewer 里能拖动观看。

前置（先起真实运行时，3D viewer 就是它开的）：
    cd microduck-main && scripts/duck-sim    # 或 by hand：duck-body + robotd --sim

运行（LV4 venv 有 jax + aif；本脚本把 microduck_rl/src 加进路径取桥/客户端）：
    cd /home/zkxt/dev/Duck
    LV4/.venv/bin/python microduck_rl/scripts/duck_pet_robotd.py \
        --socket ~/.cache/duck-sim/duck-a.sock --steps 20
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
from aif.examples import duck_pet as dp
from mjlab_microduck.duck_bridge import world_move_arc
from mjlab_microduck.robotd_client import RobotdClient

CELL = 0.5       # 每格边长（米）
ARRIVE_RADIUS = 0.2  # 距目标格中心多近算「到达」
HOME_XY = (0.0, 0.0)                     # 休息点（HOME = cell 0）
EXPLORE_XY = [(1.0, 0.0), (0.0, -1.0)]   # 探索点（cell 2 和 cell 10）


def quat_to_yaw(q: list[float]) -> float:
    w, x, y, z = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _BodyGroundTruth:
    """连 body_server（TCP）读 ground-truth trunk 位置 + 四元数——同一 MuJoCo 世界帧。"""

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
    """真实 robotd 身体：ground-truth 位置/朝向 + 精力/兴致在桥侧模拟。"""

    modality_names = dp.PetArena.modality_names

    def __init__(self, client: RobotdClient, body_addr: str = "127.0.0.1:7801", seed: int = 0):
        self.client = client
        self.energy = dp.N_ENERGY - 1
        self.fun = dp.N_FUN - 1
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
        self.client.poll_state(timeout=0.0)  # 订阅了状态流必须读，否则 deadman 阻塞

    def cell_of(self) -> int:
        c = max(0, min(dp.GRID - 1, round(self._pos[0] / CELL)))
        r = max(0, min(dp.GRID - 1, round(-self._pos[1] / CELL)))
        return r * dp.GRID + c

    def move_to_cell(self, target_cell: int, speed: float, yaw_rate: float, max_t: float = 60.0) -> None:
        tx = (target_cell % dp.GRID) * CELL
        ty = -(target_cell // dp.GRID) * CELL
        t = 0.0
        while t < max_t:
            self.refresh()
            if self.fallen:
                self.recover()
                t += 1.0
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

    def recover(self, max_wait: float = 20.0) -> bool:
        """摔倒后尝试用 sitstand 站起：先坐稳（Sitting）再站起（Rising）。"""
        self.client.move(0.0, 0.0, 0.0)
        self.client.do("sit_toggle")   # 先坐下稳住
        time.sleep(2.0)
        self.client.do("sit_toggle")   # 再站起
        t = 0.0
        while t < max_wait:
            self.refresh()
            if not self.fallen:
                return True
            time.sleep(0.5)
            t += 0.5
        return False

    def _near(self, xy: tuple[float, float], radius: float = 0.35) -> bool:
        # 半径须 ≥ ARRIVE_RADIUS(0.2)，否则鸭子「到达」格中心却还没近到触发回充。
        return math.hypot(self._pos[0] - xy[0], self._pos[1] - xy[1]) < radius

    def drain_and_recharge(self) -> None:
        self.energy = dp.N_ENERGY - 1 if self._near(HOME_XY) else max(self.energy - 1, 0)
        if any(self._near(e) for e in EXPLORE_XY):
            self.fun = dp.N_FUN - 1
        else:
            self.fun = max(self.fun - 1, 0)

    def observe(self) -> dict[str, jnp.ndarray]:
        cell = self.cell_of()
        if self._rng.random() < dp.POS_ACC:
            position = cell
        else:
            nbrs = dp._neighbors(cell)
            position = int(nbrs[self._rng.integers(len(nbrs))])
        flaky = int(self._rng.integers(dp.N_FUN))
        return {
            "position": jnp.asarray([position], dtype=jnp.int32),
            "energy": jnp.asarray([self.energy], dtype=jnp.int32),
            "fun": jnp.asarray([self.fun], dtype=jnp.int32),
            "flaky": jnp.asarray([flaky], dtype=jnp.int32),
        }

    def close(self) -> None:
        self._gt.close()


def _target_cell(cur_cell: int, action: int) -> int:
    r, c = dp._rc(cur_cell)
    nr, nc = dp._move(r, c, action)
    return dp._cell(nr, nc)


def _marker(cell: int) -> str:
    return "H" if cell == dp.HOME else ("E" if cell in dp.EXPLORE else str(cell))


def run(socket: str, steps: int, speed: float, yaw_rate: float, seed: int, body_addr: str):
    spec = dp._make_spec()
    model = dp._build_model()
    agent = SelfAwareAgent(spec, model)
    B_fun, B_boring = dp._candidate_models(model)
    agent.structure.add_model("fun", [model.B[0], model.B[1], B_fun])
    agent.structure.add_model("boring", [model.B[0], model.B[1], B_boring])

    client = RobotdClient(socket)
    client.connect()
    client.subscribe(hz=20)
    client.enable(on=True)
    body = RobotdBody(client, body_addr=body_addr, seed=seed)

    # D 先验匹配鸭子实际起点，避免硬先验与实际观测冲突腐蚀信念。
    body.refresh()
    start_cell = body.cell_of()
    agent._pymdp.D[dp.POS_FACTOR] = jnp.asarray(
        dp._onehot(dp.N_CELLS, start_cell), dtype=jnp.float32
    )[None, :]

    key = jax.random.PRNGKey(seed)
    key, k = jax.random.split(key)
    obs = body.observe()
    print(f"{'t':>3s} {'act':>3s} {'cell':>3s} {'e':>1s} {'f':>1s} {'fallen':>6s} {'γ':>6s}")
    try:
        t = 0
        while steps <= 0 or t < steps:
            key, k = jax.random.split(key)
            result = agent.step(obs_to_list(obs, body.modality_names), k)
            action = int(np.asarray(result.action).reshape(-1)[0])
            body.move_to_cell(_target_cell(body.cell_of(), action), speed, yaw_rate)
            body.drain_and_recharge()
            obs = body.observe()
            gamma = result.self_state["gamma"]
            print(f"{t:3d} {dp.ACTION_NAMES[action]:>3s} {_marker(body.cell_of()):>3s} "
                  f"{body.energy:1d} {body.fun:1d} {body.fallen!s:>6s} {gamma:6.2f}")
            t += 1
    finally:
        client.move(0.0, 0.0, 0.0)
        body.close()
        client.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="aif 宠物鸭子 ↔ robotd ↔ MuJoCo 3D 闭环")
    ap.add_argument("--socket", default="~/.cache/duck-sim/duck-a.sock")
    ap.add_argument("--body", default="127.0.0.1:7801", help="duck-body 的 host:port（读 ground-truth）")
    ap.add_argument("--steps", type=int, default=0, help="步数上限，0 = 无限一直跑")
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--yaw-rate", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(args.socket, args.steps, args.speed, args.yaw_rate, args.seed, args.body)


if __name__ == "__main__":
    main()
