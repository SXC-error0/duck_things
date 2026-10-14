"""robotd IPC 客户端：纯 Python 的 newline-delimited JSON-RPC。

这是「aif 认知层 ↔ 真实 robotd 守护进程」的接缝（Option 1）。认知层不直接碰 MuJoCo /
ONNX，而是像 gamepad（``padd``）一样，作为 ``robotd`` 的 IPC 客户端：发 ``robot.move``
速度意图、读 ``robot.state`` 状态流（摔倒 / 里程计 / IMU）。摔倒检测、standup、策略热插拔
全由 robotd 自己负责——这正是真实鸭子运行时已有的兜底，绕开了「单独 velocity 策略走不稳」
的 sim2real 坑。

协议（与 ``duck-ipc-proto`` 对齐，见 microduck-main 文档）：

- 帧：换行分隔的 JSON，每行一条消息。
- 请求（需要应答）：``{"jsonrpc":"2.0","id":N,"method":...,"params":{...}}``。
- 通知（无需应答）：同上但**无 ``id``**——连续意图（``robot.move``）走这条。
- 状态流：订阅 ``robot.subscribe`` 后，robotd 逐 tick 推送 ``robot.state`` 通知。

关键方法（本模块覆盖认知层需要的）：

- ``robot.move {vx, vy, vyaw}``   速度 twist（前进/左移 m/s，左转 rad/s），通知。
- ``robot.stop {}``               零速度。
- ``robot.enable {on}``           启用/停用策略（→ ``IntentResult``）。
- ``robot.do {skill}``            一次性技能（踢球等，→ ``IntentResult``）。
- ``robot.subscribe {hz?}``       订阅状态流（→ ``SubscribeResult``）。
- ``robot.state``（通知）         逐 tick 状态，含 ``safety.fallen`` / ``odom.position`` /
                                  ``imu.quat``。

纯标准库，无 jax / mujoco / numpy 依赖。
"""

from __future__ import annotations

import json
import select
import socket
from dataclasses import dataclass
from typing import Any


class RobotdError(Exception):
    """robotd 返回的 JSON-RPC 错误。"""


@dataclass
class State:
    """一次 ``robot.state`` 里认知层关心的字段（其余原样保留在 ``raw``）。"""

    fallen: bool
    odom: list[float]        # [x, y, z]（接触里程计，默认零）
    quat: list[float] | None  # [w, x, y, z]（trunk 姿态，缺省 None）
    policy: str
    raw: dict[str, Any]

    @classmethod
    def from_msg(cls, params: dict[str, Any]) -> State:
        safety = params.get("safety", {}) or {}
        odom = params.get("odom", {}) or {}
        imu = params.get("imu") or {}
        return cls(
            fallen=bool(safety.get("fallen", False)),
            odom=list(odom.get("position", [0.0, 0.0, 0.0])),
            quat=list(imu["quat"]) if imu.get("quat") is not None else None,
            policy=str(params.get("policy", "")),
            raw=params,
        )


class RobotdClient:
    """robotd 的 IPC 客户端。``addr`` 为 Unix socket 路径或 ``"host:port"``。"""

    def __init__(self, addr: str, timeout: float = 2.0):
        self.addr = addr
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._buf = b""
        self._next_id = 1
        self._latest_state: State | None = None

    # -- 连接 -- -------------------------------------------------------------

    def connect(self) -> None:
        if ":" in self.addr and not self.addr.startswith("/"):
            host, port = self.addr.rsplit(":", 1)
            self._sock = socket.create_connection((host, int(port)), timeout=self.timeout)
        else:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._sock.connect(self.addr)

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    # -- 底层收发 -- ---------------------------------------------------------

    def _send(self, obj: dict[str, Any]) -> None:
        if self._sock is None:
            raise RuntimeError("not connected")
        self._sock.sendall((json.dumps(obj) + "\n").encode())

    def _read_line(self) -> dict[str, Any]:
        """阻塞读一条 JSON 行。"""
        if self._sock is None:
            raise RuntimeError("not connected")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("robotd 连接已关闭")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return json.loads(line.decode())

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        msg: dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """带 id 的请求，读到匹配应答为止；途中把状态通知存进 ``_latest_state``。"""
        msg_id = self._next_id
        self._next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)
        while True:
            reply = self._read_line()
            if reply.get("id") == msg_id:
                if "error" in reply:
                    raise RobotdError(reply["error"])
                return reply.get("result", {})
            if reply.get("method") == "robot.state" and "params" in reply:
                self._latest_state = State.from_msg(reply["params"])

    def poll_state(self, timeout: float = 0.0) -> State | None:
        """读状态流：先排空缓冲，再（最多 ``timeout`` 秒）等待新状态；返回最新状态或 None。"""
        if self._sock is None:
            return self._latest_state
        self._drain_state_lines()
        if timeout > 0:
            ready, _, _ = select.select([self._sock], [], [], timeout)
            if not ready:
                return self._latest_state
        try:
            self._sock.setblocking(False)
            chunk = self._sock.recv(65536)
        except BlockingIOError:
            chunk = b""
        finally:
            self._sock.setblocking(True)
        if chunk:
            self._buf += chunk
            self._drain_state_lines()
        return self._latest_state

    def _drain_state_lines(self) -> None:
        """把缓冲里完整的行解析出来，状态通知存进 ``_latest_state``。"""
        while b"\n" in self._buf:
            line, self._buf = self._buf.split(b"\n", 1)
            msg = json.loads(line.decode())
            if msg.get("method") == "robot.state" and "params" in msg:
                self._latest_state = State.from_msg(msg["params"])

    # -- 认知层关心的便捷方法 -- ---------------------------------------------

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        self.notify("robot.move", {"vx": vx, "vy": vy, "vyaw": vyaw})

    def stop(self) -> None:
        self.notify("robot.stop", {})

    def enable(self, on: bool = True) -> dict[str, Any]:
        return self.call("robot.enable", {"on": on})

    def do(self, skill: str) -> dict[str, Any]:
        return self.call("robot.do", {"skill": skill})

    def subscribe(self, hz: int | None = None) -> dict[str, Any]:
        params = {"hz": hz} if hz is not None else {}
        return self.call("robot.subscribe", params)
