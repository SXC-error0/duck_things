"""robotd IPC 客户端（``mjlab_microduck.robotd_client``）的单元测试。

用一个 in-process 的 mock robotd（线程 + TCP）验证：帧格式、move 通知、subscribe 应答、
状态流读取。锁定协议与 ``duck-ipc-proto`` 对齐。
"""

from __future__ import annotations

import json
import queue
import socket
import threading

import pytest

from mjlab_microduck.robotd_client import RobotdClient, State


class MockRobotd:
    """极简 robotd：记录请求，对 call 应答，对 subscribe 后推一次状态。"""

    def __init__(self):
        self.requests: queue.Queue = queue.Queue()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=2.0)

    def _serve(self):
        conn, _ = self._sock.accept()
        conn.settimeout(0.2)
        buf = b""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(65536)
            except TimeoutError:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = json.loads(line.decode())
                self.requests.put(msg)
                self._reply(conn, msg)
        conn.close()

    def _reply(self, conn, msg):
        mid = msg.get("id")
        method = msg.get("method")
        if mid is None:
            return  # 通知（move/stop），不应答
        if method == "robot.subscribe":
            conn.sendall((json.dumps({
                "jsonrpc": "2.0", "id": mid,
                "result": {"accepted": True, "walk": "walk.onnx", "skills": ["kick"]},
            }) + "\n").encode())
            # 订阅后推一帧状态，供 poll_state 读取。
            conn.sendall((json.dumps({
                "jsonrpc": "2.0", "method": "robot.state",
                "params": {
                    "t": 1.0, "move": {}, "head": [0, 0, 0, 0], "policy": "walk",
                    "safety": {"fallen": False},
                    "odom": {"position": [0.5, -0.3, 0.0]},
                    "imu": {"gyro": [0, 0, 0], "quat": [1.0, 0.0, 0.0, 0.0]},
                    "joints": [0.0] * 15, "targets": [0.0] * 15,
                },
            }) + "\n").encode())
        else:
            conn.sendall((json.dumps({
                "jsonrpc": "2.0", "id": mid, "result": {"accepted": True},
            }) + "\n").encode())


@pytest.fixture()
def mock_robotd():
    m = MockRobotd()
    m.start()
    yield m
    m.stop()


def test_move_sends_notification(mock_robotd):
    client = RobotdClient(f"127.0.0.1:{mock_robotd.port}")
    client.connect()
    client.move(0.3, -0.1, 0.5)
    client.close()

    msg = mock_robotd.requests.get(timeout=1.0)
    assert msg["method"] == "robot.move"
    assert msg["jsonrpc"] == "2.0"
    assert "id" not in msg  # 连续意图是通知
    assert msg["params"] == {"vx": 0.3, "vy": -0.1, "vyaw": 0.5}


def test_subscribe_returns_result_and_state(mock_robotd):
    client = RobotdClient(f"127.0.0.1:{mock_robotd.port}")
    client.connect()
    result = client.subscribe(hz=10)
    assert result["accepted"] is True
    assert result["walk"] == "walk.onnx"
    assert result["skills"] == ["kick"]

    state = client.poll_state(timeout=0.5)
    assert state is not None
    assert isinstance(state, State)
    assert state.fallen is False
    assert state.odom == [0.5, -0.3, 0.0]
    assert state.quat == [1.0, 0.0, 0.0, 0.0]
    assert state.policy == "walk"
    client.close()


def test_enable_and_do_are_calls(mock_robotd):
    client = RobotdClient(f"127.0.0.1:{mock_robotd.port}")
    client.connect()
    assert client.enable(on=True)["accepted"] is True
    assert client.do("kick")["accepted"] is True
    client.close()

    # 两者都是带 id 的 call（应有应答）。
    msgs = [mock_robotd.requests.get(timeout=1.0) for _ in range(2)]
    methods = {m["method"] for m in msgs}
    assert methods == {"robot.enable", "robot.do"}
    for m in msgs:
        assert "id" in m
