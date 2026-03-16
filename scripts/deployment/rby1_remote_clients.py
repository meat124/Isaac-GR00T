from __future__ import annotations

from typing import Any

import zmq

from gr00t.policy.server_client import MsgSerializer, PolicyClient


class ObservationClient:
    """Client for rby1_observation_runner.py."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5556,
        timeout_ms: int = 15000,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_ms = int(timeout_ms)

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self.host}:{self.port}")
        self._socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint, "data": data or {}}
        self._socket.send(MsgSerializer.to_bytes(request))
        message = self._socket.recv()
        response = MsgSerializer.from_bytes(message)
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"Observation runner error: {response['error']}")
        return response

    def ping(self) -> bool:
        try:
            self._call("ping")
            return True
        except Exception:
            return False

    def get_status(self) -> dict[str, Any]:
        return self._call("get_status")

    def get_observation(self) -> dict[str, Any]:
        return self._call("get_observation")

    def close_runner(self) -> dict[str, Any]:
        return self._call("close")

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class Gr00tRemotePolicyClient(PolicyClient):
    """Named wrapper for clarity in notebook code paths."""

    pass
