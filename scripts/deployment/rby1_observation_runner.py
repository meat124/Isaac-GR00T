from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import tyro
import yaml
import zmq

from gr00t.policy.server_client import MsgSerializer


@dataclass
class Args:
    """Run an external RBY1 observation service (camera + state only)."""

    host: str = "0.0.0.0"
    port: int = 5556
    config_yaml: str = "/home/hyunjin/rby1_ws/rby1-data-collection/config.yaml"

    robot_ip: str | None = None
    prompt: str = "pick up the cup and place it on the plate"

    cam_head_serial: str | None = None
    cam_left_serial: str | None = None
    cam_right_serial: str | None = None

    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    render_width: int = 224
    render_height: int = 224

    arm_command_priority: int = 10
    arm_minimum_time: float = 10.0
    log_action_send: bool = False

    use_remote_gripper: bool = True
    state_source: str = "robot"


class ObservationRunner:
    def __init__(self, args: Args):
        self.args = args
        self._running = True
        self._env = None
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://{args.host}:{args.port}")

    def _load_defaults_from_yaml(self) -> dict[str, Any]:
        cfg_path = Path(self.args.config_yaml).expanduser().resolve()
        if not cfg_path.exists():
            return {}
        with cfg_path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cameras = raw.get("cameras", {})
        img_size = raw.get("img_size", {})
        return {
            "robot_ip": raw.get("user_pc_ip"),
            "cam_head_serial": cameras.get("head", {}).get("serial"),
            "cam_left_serial": cameras.get("left", {}).get("serial"),
            "cam_right_serial": cameras.get("right", {}).get("serial"),
            "camera_width": img_size.get("width"),
            "camera_height": img_size.get("height"),
        }

    def _resolve_runtime_config(self) -> dict[str, Any]:
        defaults = self._load_defaults_from_yaml()

        robot_ip = self.args.robot_ip or defaults.get("robot_ip")
        if not robot_ip:
            raise ValueError("robot_ip is required (pass --robot-ip or set user_pc_ip in config.yaml)")

        cam_head_serial = self.args.cam_head_serial or defaults.get("cam_head_serial")
        cam_left_serial = self.args.cam_left_serial or defaults.get("cam_left_serial")
        cam_right_serial = self.args.cam_right_serial or defaults.get("cam_right_serial")
        if not (cam_head_serial and cam_left_serial and cam_right_serial):
            raise ValueError("camera serials are required (args or config.yaml)")

        camera_width = self.args.camera_width or defaults.get("camera_width")
        camera_height = self.args.camera_height or defaults.get("camera_height")
        if camera_width is None or camera_height is None:
            raise ValueError("camera width/height are required (args or config.yaml)")

        return {
            "robot_ip": robot_ip,
            "cam_head_serial": cam_head_serial,
            "cam_left_serial": cam_left_serial,
            "cam_right_serial": cam_right_serial,
            "camera_width": int(camera_width),
            "camera_height": int(camera_height),
        }

    def _init_env(self) -> None:
        try:
            from scripts.deployment import rby1_env as _env  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError:
            import rby1_env as _env  # pylint: disable=import-outside-toplevel

        cfg = self._resolve_runtime_config()
        logging.info("Observation config: %s", cfg)

        self._env = _env.RBY1Environment(
            robot_ip=cfg["robot_ip"],
            prompt=self.args.prompt,
            render_height=self.args.render_height,
            render_width=self.args.render_width,
            camera_width=cfg["camera_width"],
            camera_height=cfg["camera_height"],
            camera_fps=self.args.camera_fps,
            cam_head_serial=cfg["cam_head_serial"],
            cam_left_serial=cfg["cam_left_serial"],
            cam_right_serial=cfg["cam_right_serial"],
            left_action_dim=8,
            right_action_dim=8,
            arm_command_priority=self.args.arm_command_priority,
            arm_minimum_time=self.args.arm_minimum_time,
            log_action_send=self.args.log_action_send,
            state_source=self.args.state_source,
            state_zmq_address=None,
            state_indices=None,
            gripper_state_key=None,
            use_remote_gripper=self.args.use_remote_gripper,
            gripper=None,
        )

    def _handle_ping(self) -> dict[str, Any]:
        return {"status": "ok", "message": "observation runner is running"}

    def _handle_get_status(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "runner_type": "rby1_observation_runner",
            "env_ready": self._env is not None,
            "host": self.args.host,
            "port": self.args.port,
        }

    def _handle_get_observation(self) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("Environment is not initialized")
        return self._env.get_observation()

    def _handle_close(self) -> dict[str, Any]:
        self._running = False
        return {"status": "ok", "message": "runner shutting down"}

    def _dispatch(self, endpoint: str) -> Any:
        if endpoint == "ping":
            return self._handle_ping()
        if endpoint == "get_status":
            return self._handle_get_status()
        if endpoint == "get_observation":
            return self._handle_get_observation()
        if endpoint == "close":
            return self._handle_close()
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def run(self) -> None:
        logging.info("Initializing RBY1 observation environment...")
        self._init_env()
        addr = self._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        logging.info("Observation runner listening on %s", addr)

        while self._running:
            try:
                message = self._socket.recv()
                request = MsgSerializer.from_bytes(message)
                endpoint = request.get("endpoint", "ping")
                result = self._dispatch(endpoint)
                self._socket.send(MsgSerializer.to_bytes(result))
            except Exception as exc:  # noqa: BLE001
                logging.exception("Observation runner error")
                self._socket.send(MsgSerializer.to_bytes({"error": str(exc)}))

        self.close()

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception as exc:  # noqa: BLE001
                logging.warning("Failed to close env cleanly: %s", exc)
            self._env = None
        self._socket.close(0)
        self._context.term()


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    runner = ObservationRunner(args)
    try:
        runner.run()
    except KeyboardInterrupt:
        logging.info("Interrupted. Closing runner...")
        runner.close()


if __name__ == "__main__":
    main(tyro.cli(Args))
