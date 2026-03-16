from __future__ import annotations

import logging
import pickle
import time
from typing import Optional, Sequence

import numpy as np

try:
    from scripts.deployment.rby1_remote_gripper import Gripper
except ModuleNotFoundError:
    from rby1_remote_gripper import Gripper

try:
    import pyrealsense2 as rs
except ImportError as exc:  # pragma: no cover
    raise ImportError("pyrealsense2 is required for RBY1Environment") from exc

try:
    import rby1_sdk as rby
except ImportError as exc:  # pragma: no cover
    raise ImportError("rby1_sdk is required for RBY1Environment") from exc

try:
    import zmq
except ImportError:  # pragma: no cover
    zmq = None


logger = logging.getLogger(__name__)


def _convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    if np.issubdtype(img.dtype, np.floating):
        img = np.clip(img, 0.0, 1.0) * 255.0
    else:
        img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def _resize_nearest(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = image.shape[:2]
    y_idx = (np.arange(out_h) * (in_h / out_h)).astype(int)
    x_idx = (np.arange(out_w) * (in_w / out_w)).astype(int)
    y_idx = np.clip(y_idx, 0, in_h - 1)
    x_idx = np.clip(x_idx, 0, in_w - 1)
    return image[y_idx[:, None], x_idx]


def _resize_with_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    in_h, in_w = image.shape[:2]
    if in_h <= 0 or in_w <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")

    scale = min(target_h / in_h, target_w / in_w)
    new_h = max(1, int(round(in_h * scale)))
    new_w = max(1, int(round(in_w * scale)))

    resized = _resize_nearest(image, new_h, new_w)
    out = np.zeros((target_h, target_w, image.shape[2]), dtype=resized.dtype)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    out[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return out


class _RealsenseCamera:
    def __init__(
        self,
        *,
        serial: Optional[str],
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self._serial = serial
        self._width = width
        self._height = height
        self._fps = fps
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        if self._serial is not None:
            self._config.enable_device(self._serial)
        self._config.enable_stream(rs.stream.color, self._width, self._height, rs.format.rgb8, self._fps)
        self._started = False

    def __del__(self) -> None:
        self.stop()

    def start(self) -> None:
        if self._started:
            return
        connected_serials = []
        try:
            ctx = rs.context()
            for dev in ctx.query_devices():
                try:
                    connected_serials.append(dev.get_info(rs.camera_info.serial_number))
                except Exception:  # noqa: BLE001
                    continue
            del ctx
        except Exception:  # noqa: BLE001
            connected_serials = []

        if self._serial is not None and connected_serials and self._serial not in connected_serials:
            raise RuntimeError(
                f"Camera serial {self._serial} not found. Connected RealSense serials: {connected_serials}"
            )

        try:
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._pipeline.start(self._config)
            self._started = True
            for _ in range(30):
                self._pipeline.wait_for_frames(timeout_ms=1000)
        except Exception as exc:  # noqa: BLE001
            try:
                self._pipeline.stop()
            except Exception:  # noqa: BLE001
                pass
            self._started = False
            raise RuntimeError(f"Failed to start camera {self._serial}: {exc}") from exc

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self._pipeline.stop()
        except Exception:  # noqa: BLE001
            pass
        self._started = False

    def get_rgb_image(self) -> np.ndarray:
        if not self._started:
            self.start()
        if not self._started:
            raise RuntimeError(f"Camera {self._serial} is not started")

        for attempt in range(3):
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=3000)
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    raise RuntimeError("Received frames but no color frame found")
                return np.asanyarray(color_frame.get_data())
            except RuntimeError as exc:
                if "cannot be called before start" in str(exc).lower():
                    raise RuntimeError(f"Camera {self._serial} is not running") from exc
                if attempt < 2:
                    self._started = False
                    try:
                        self._pipeline.stop()
                    except RuntimeError:
                        pass
                    self.start()
                else:
                    raise

        raise RuntimeError("Failed to get image after retries")


class RBY1Environment:
    """RBY1 environment used by GR00T notebook without openpi package imports."""

    def __init__(
        self,
        *,
        robot_ip: str,
        prompt: str = "pick up the object",
        render_height: int = 224,
        render_width: int = 224,
        camera_width: int = 640,
        camera_height: int = 480,
        camera_fps: int = 30,
        cam_head_serial: Optional[str] = None,
        cam_left_serial: Optional[str] = None,
        cam_right_serial: Optional[str] = None,
        left_action_dim: int = 8,
        right_action_dim: int = 8,
        arm_command_priority: int = 1,
        arm_action_scale: float = 1.0,
        arm_minimum_time: float = 0.1,
        log_action_send: bool = False,
        state_source: str = "robot",
        state_zmq_address: Optional[str] = None,
        state_indices: Optional[Sequence[int]] = None,
        gripper_state_key: Optional[str] = None,
        use_remote_gripper: bool = True,
        gripper: Optional[object] = None,
        robot: Optional[object] = None,
    ) -> None:
        self._prompt = prompt
        self._render_height = render_height
        self._render_width = render_width
        self._left_action_dim = left_action_dim
        self._right_action_dim = right_action_dim
        self._arm_command_priority = int(arm_command_priority)
        self._arm_action_scale = arm_action_scale
        self._arm_minimum_time = arm_minimum_time
        self._log_action_send = bool(log_action_send)
        self._state_source = state_source
        self._state_indices = np.asarray(state_indices, dtype=int) if state_indices is not None else None
        self._gripper_state_key = gripper_state_key
        self._use_remote_gripper = use_remote_gripper
        self._gripper = gripper

        self._cameras = {
            "observation/head_image": _RealsenseCamera(
                serial=cam_head_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "observation/left_wrist_image": _RealsenseCamera(
                serial=cam_left_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
            "observation/right_wrist_image": _RealsenseCamera(
                serial=cam_right_serial,
                width=camera_width,
                height=camera_height,
                fps=camera_fps,
            ),
        }

        for cam in self._cameras.values():
            cam.start()

        self._robot = robot if robot is not None else self._create_robot(robot_ip)
        self._robot.connect()
        if not self._robot.is_connected():
            raise RuntimeError("Failed to connect to robot")

        self._prepare_robot_for_control()

        self._state_socket = None
        if self._state_source == "zmq":
            if zmq is None:
                raise ImportError("pyzmq is required when state_source='zmq'")
            if not state_zmq_address:
                raise ValueError("state_zmq_address is required when state_source='zmq'")
            context = zmq.Context.instance()
            socket = context.socket(zmq.SUB)
            socket.connect(state_zmq_address)
            socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._state_socket = socket

        if self._use_remote_gripper and self._gripper is None:
            self._gripper = Gripper()
            try:
                if self._gripper.initialize(verbose=True) and self._gripper.homing():
                    self._gripper.start()
                    self._gripper.set_normalized_target(np.array([1.0, 1.0]))
                else:
                    self._gripper = None
            except Exception:  # noqa: BLE001
                self._gripper = None

        if self._use_remote_gripper and self._gripper is None:
            raise RuntimeError(
                "Remote gripper is required but unavailable. "
                "Run gripper init first or set use_remote_gripper=False."
            )

    def _create_robot(self, robot_ip: str) -> object:
        if hasattr(rby, "create_robot"):
            return rby.create_robot(robot_ip, "a")
        raise RuntimeError("Unable to construct RBY1 robot client from rby1_sdk")

    def _prepare_robot_for_control(self) -> None:
        try:
            if hasattr(self._robot, "power_on"):
                self._robot.power_on(".*")
            if hasattr(self._robot, "servo_on"):
                self._robot.servo_on(".*")
            if hasattr(self._robot, "enable_control_manager"):
                self._robot.enable_control_manager()
            if hasattr(self._robot, "cancel_control"):
                try:
                    self._robot.cancel_control()
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(self._robot, "wait_for_control_ready"):
                self._robot.wait_for_control_ready(1000)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to fully prepare robot control manager: %s", exc)

    def reset(self) -> None:
        if hasattr(self._robot, "reset"):
            self._robot.reset()

    def is_episode_complete(self) -> bool:
        return False

    def get_observation(self) -> dict:
        observation = {}
        for name, camera in self._cameras.items():
            raw_img = camera.get_rgb_image()
            resized = _resize_with_pad(raw_img, self._render_height, self._render_width)
            resized = _convert_to_uint8(resized)
            observation[name] = np.transpose(resized, (2, 0, 1))

        observation["observation/state"] = self._get_joint_positions()
        observation["prompt"] = self._prompt
        return observation

    def _get_joint_positions(self) -> np.ndarray:
        if self._state_source == "zmq":
            return self._get_joint_positions_from_zmq()

        if not hasattr(self._robot, "get_state"):
            raise RuntimeError("rby1_sdk robot object must provide get_state()")
        qpos = self._robot.get_state().position
        qpos = np.asarray(qpos[8:22], dtype=np.float32).reshape(-1)
        qpos = self._append_gripper_state(qpos)
        return self._apply_state_indices(qpos)

    def _get_joint_positions_from_zmq(self) -> np.ndarray:
        if self._state_socket is None:
            raise RuntimeError("ZMQ state socket is not initialized")

        msg = self._state_socket.recv()
        data = pickle.loads(msg)
        qpos = self._extract_state_field(data, "joint_positions")
        if qpos is None:
            raise RuntimeError("ZMQ state does not contain joint_positions")

        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if self._gripper_state_key is not None:
            gripper = self._extract_state_field(data, self._gripper_state_key)
            if gripper is not None:
                gripper = np.asarray(gripper, dtype=np.float32).reshape(-1)
                qpos = np.concatenate([qpos, gripper], axis=0)
        else:
            qpos = self._append_gripper_state(qpos)
        return self._apply_state_indices(qpos)

    def _append_gripper_state(self, qpos: np.ndarray) -> np.ndarray:
        if self._gripper is None:
            if self._use_remote_gripper:
                raise RuntimeError(
                    "Remote gripper is enabled but gripper client is not initialized"
                )
            return qpos
        try:
            gripper_state = np.asarray(self._gripper.get_state(), dtype=np.float32).reshape(-1)
            if gripper_state.size != 2:
                raise RuntimeError(f"Invalid remote gripper state shape: {gripper_state.shape}")
            return np.concatenate([qpos, gripper_state], axis=0)
        except Exception as exc:  # noqa: BLE001
            if self._use_remote_gripper:
                raise RuntimeError(f"Failed to fetch remote gripper state: {exc}") from exc
            logger.warning("Failed to fetch remote gripper state: %s", exc)
            return qpos

    @staticmethod
    def _extract_state_field(data, key: str):
        if isinstance(data, dict):
            return data.get(key)
        return getattr(data, key, None)

    def _apply_state_indices(self, qpos: np.ndarray) -> np.ndarray:
        if self._state_indices is None:
            return qpos
        return qpos[self._state_indices]

    def apply_action(self, action: dict) -> None:
        if "actions" not in action:
            raise KeyError("Action dict missing 'actions' key")

        action_vec = np.asarray(action["actions"], dtype=np.float32).reshape(-1)
        expected = self._left_action_dim + self._right_action_dim
        if action_vec.size != expected:
            raise ValueError(f"Action dimension mismatch (expected {expected}, got {action_vec.size})")

        right_action = action_vec[: self._right_action_dim]
        left_action = action_vec[self._right_action_dim : expected]
        self._send_joint_positions(left_action, right_action)

    def _send_joint_positions(self, left_action: np.ndarray, right_action: np.ndarray) -> None:
        left_arm = left_action[:7] if left_action.size >= 7 else left_action
        right_arm = right_action[:7] if right_action.size >= 7 else right_action
        left_gripper = left_action[7] if left_action.size > 7 else None
        right_gripper = right_action[7] if right_action.size > 7 else None

        if self._gripper is not None and (left_gripper is not None or right_gripper is not None):
            try:
                gripper_target = self._gripper.get_target()
                if right_gripper is not None:
                    gripper_target[0] = float(right_gripper)
                if left_gripper is not None:
                    gripper_target[1] = float(left_gripper)
                self._gripper.set_normalized_target(gripper_target, wait_for_reply=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to send gripper command: %s", exc)

        rv = self._robot.send_command(
            rby.RobotCommandBuilder().set_command(
                rby.ComponentBasedCommandBuilder().set_body_command(
                    rby.BodyComponentBasedCommandBuilder()
                    .set_right_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(
                            rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time))
                        )
                        .set_position(self._arm_action_scale * right_arm)
                        .set_minimum_time(self._arm_minimum_time)
                    )
                    .set_left_arm_command(
                        rby.JointPositionCommandBuilder()
                        .set_command_header(
                            rby.CommandHeaderBuilder().set_control_hold_time(max(0.2, self._arm_minimum_time))
                        )
                        .set_position(self._arm_action_scale * left_arm)
                        .set_minimum_time(self._arm_minimum_time)
                    )
                )
            ),
            self._arm_command_priority,
        ).get()

        finish_code = getattr(rv, "finish_code", None)
        finish_enum = getattr(getattr(rby, "RobotCommandFeedback", None), "FinishCode", None)
        if finish_enum is not None and finish_code is not None and finish_code != finish_enum.Ok:
            logger.warning("Robot arm command finish_code=%s", finish_code)

        if self._log_action_send and hasattr(self._robot, "get_state"):
            try:
                q_before = np.asarray(self._robot.get_state().position, dtype=np.float64)[8:22].copy()
                time.sleep(max(0.02, min(0.2, float(self._arm_minimum_time))))
                q_after = np.asarray(self._robot.get_state().position, dtype=np.float64)[8:22].copy()
                dq = q_after - q_before
                logger.info("[send] q_delta_norm=%.6f", float(np.linalg.norm(dq)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("[send] failed to read q after command: %s", exc)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        if self._gripper is not None:
            try:
                self._gripper.stop()
            except Exception:  # noqa: BLE001
                pass

        if hasattr(self._robot, "disconnect"):
            try:
                self._robot.disconnect()
            except Exception:  # noqa: BLE001
                pass

        for camera in self._cameras.values():
            camera.stop()
