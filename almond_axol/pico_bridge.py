"""Bridge PICO XRoboToolkit tracking into Axol's existing VRFrame protocol.

The bridge intentionally does not import Axol's VR server classes. It is a
client that connects to the already-running ``VRServer`` and sends the same
JSON shape as the WebXR app. This keeps the IK, teleop, collect-data, and
robot-control paths unchanged.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import ssl
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.spatial.transform import Rotation

_logger = logging.getLogger(__name__)


PicoEeSource = Literal["controller", "wrist", "hand"]
PicoCoordinateMode = Literal["unity", "webxr", "axol-vr", "body"]
PicoOrientationMode = Literal["tracking", "controller", "identity"]
PicoFreshnessMode = Literal["timestamp", "local"]
PicoGripperMode = Literal["invert-trigger", "direct-trigger", "constant-open"]
PicoElbowSource = Literal["body", "frozen", "synthetic"]


@dataclass
class PicoBridgeConfig:
    """Runtime config for the PICO XRoboToolkit -> Axol bridge."""

    host: str = "localhost"
    port: int = 8000
    path: str = "/ws"
    tls: bool = True
    verify_tls: bool = False
    frequency: float = 90.0
    ee_source: PicoEeSource = "controller"
    coordinate_mode: PicoCoordinateMode = "unity"
    orientation_mode: PicoOrientationMode = "tracking"
    position_scale: float = 1.0
    freshness_mode: PicoFreshnessMode = "timestamp"
    gripper_mode: PicoGripperMode = "invert-trigger"
    elbow_source: PicoElbowSource = "body"
    body_forward_sign: float = 1.0
    grip_deadzone: float = 0.02
    lock_threshold: float = 0.5
    trigger_deadzone: float = 0.02
    auto_engage: bool = False
    record_countdown_s: float = 3.0
    stale_timeout_s: float = 0.25
    wait_body_timeout_s: float = 60.0
    service_script: str = "/opt/apps/roboticsservice/runService.sh"
    start_service: bool = True
    dry_run: bool = False
    log_every_s: float = 2.0


class _AxolState:
    TELEOP = "teleop"
    DATA_COLLECTION = "data_collection"
    RECORDING = "recording"
    SAVING = "saving"
    ERROR = "error"


_BODY_LEFT_ELBOW = 18
_BODY_RIGHT_ELBOW = 19
_BODY_LEFT_WRIST = 20
_BODY_RIGHT_WRIST = 21
_BODY_LEFT_HAND = 22
_BODY_RIGHT_HAND = 23
_BODY_PELVIS = 0
_BODY_NECK = 12
_BODY_LEFT_SHOULDER = 16
_BODY_RIGHT_SHOULDER = 17

# XRoboToolkit body/controller poses are often Unity-like in practice:
#   X right, Y up, Z forward.
# Axol's VRFrame is interpreted by IKWorker as:
#   X down, Y left, Z forward.
# Position map: (x_right, y_up, z_fwd) -> (x_down, y_left, z_fwd)
#             = (-y_up, -x_right, z_fwd)
_UNITY_TO_AXOL_VR = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

# WebXR reference spaces are typically:
#   X right, Y up, -Z forward.
_WEBXR_TO_AXOL_VR = np.array(
    [
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ],
    dtype=np.float64,
)

_VR_TO_FLU = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def _load_xrt() -> Any:
    try:
        import xrobotoolkit_sdk as xrt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "xrobotoolkit_sdk is not importable. Install XRoboToolkit PC Service "
            "bindings first, or run this bridge from the GR00T .venv_teleop "
            "environment with websockets/scipy available."
        ) from exc
    return xrt


def _maybe_start_service(config: PicoBridgeConfig) -> subprocess.Popen[Any] | None:
    if not config.start_service:
        return None
    if not config.service_script:
        return None
    if not os.path.exists(config.service_script):
        _logger.warning("XRoboToolkit service script not found: %s", config.service_script)
        return None
    _logger.info("starting XRoboToolkit service: %s", config.service_script)
    return subprocess.Popen(
        ["bash", config.service_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _button(xrt: Any, name: str, default: bool = False) -> bool:
    try:
        return bool(getattr(xrt, name)())
    except Exception:
        return default


def _analog(xrt: Any, name: str, default: float = 0.0) -> float:
    try:
        return _as_float(getattr(xrt, name)(), default)
    except Exception:
        return default


def _apply_deadzone(value: float, deadzone: float) -> float:
    if abs(value) < deadzone:
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def _gripper_value(trigger: float, mode: PicoGripperMode) -> float:
    if mode == "constant-open":
        return 1.0
    if mode == "direct-trigger":
        return float(np.clip(trigger, 0.0, 1.0))
    return float(1.0 - np.clip(trigger, 0.0, 1.0))


def _normalize_vec(value: np.ndarray, label: str) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm <= 1e-9:
        raise ValueError(f"cannot normalize near-zero {label} vector")
    return value / norm


def _pose_array(raw: Any, label: str) -> np.ndarray:
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 7:
        raise ValueError(f"{label} pose must have at least 7 values, got {arr!r}")
    quat = arr[3:7]
    norm = np.linalg.norm(quat)
    if norm <= 1e-9:
        raise ValueError(f"{label} quaternion has near-zero norm")
    arr = arr[:7].copy()
    arr[3:7] = quat / norm
    return arr


def _basis_for_mode(mode: PicoCoordinateMode) -> np.ndarray | None:
    if mode == "axol-vr":
        return None
    if mode == "body":
        raise ValueError("body coordinate mode requires a calibrated body basis")
    if mode == "webxr":
        return _WEBXR_TO_AXOL_VR
    return _UNITY_TO_AXOL_VR


def _flu_position_to_vr(pos_flu: np.ndarray) -> np.ndarray:
    return _VR_TO_FLU.T @ pos_flu


def _flu_matrix_to_vr_quat(rot_flu: np.ndarray) -> np.ndarray:
    rot_vr = _VR_TO_FLU.T @ rot_flu @ _VR_TO_FLU
    quat = Rotation.from_matrix(rot_vr).as_quat()
    return quat / np.linalg.norm(quat)


def _convert_position(
    pos: np.ndarray,
    mode: PicoCoordinateMode,
    position_scale: float,
    body_raw_to_flu: np.ndarray | None = None,
    body_origin_raw: np.ndarray | None = None,
) -> np.ndarray:
    if mode == "body":
        if body_raw_to_flu is None or body_origin_raw is None:
            raise ValueError("body coordinate mode requires calibration")
        flu = body_raw_to_flu @ (pos - body_origin_raw)
        return _flu_position_to_vr(flu) * position_scale
    basis = _basis_for_mode(mode)
    out = pos.astype(np.float64) if basis is None else basis @ pos
    return out * position_scale


def _convert_quat_xyzw(
    quat_xyzw: np.ndarray,
    mode: PicoCoordinateMode,
    orientation_mode: PicoOrientationMode,
    body_raw_to_flu: np.ndarray | None = None,
) -> np.ndarray:
    if orientation_mode == "identity":
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quat_xyzw = quat_xyzw / np.linalg.norm(quat_xyzw)
    if mode == "body":
        if body_raw_to_flu is None:
            raise ValueError("body coordinate mode requires calibration")
        rot_raw = Rotation.from_quat(quat_xyzw).as_matrix()
        rot_flu = body_raw_to_flu @ rot_raw @ body_raw_to_flu.T
        return _flu_matrix_to_vr_quat(rot_flu)
    basis = _basis_for_mode(mode)
    if basis is None:
        return quat_xyzw.astype(np.float64)
    rot = Rotation.from_quat(quat_xyzw).as_matrix()
    converted = basis @ rot @ basis.T
    out = Rotation.from_matrix(converted).as_quat()
    return out / np.linalg.norm(out)


def _pose_to_vr_json(
    pose: np.ndarray,
    mode: PicoCoordinateMode,
    orientation_mode: PicoOrientationMode,
    position_scale: float,
    body_raw_to_flu: np.ndarray | None = None,
    body_origin_raw: np.ndarray | None = None,
) -> dict[str, Any]:
    pos = _convert_position(
        pose[:3],
        mode,
        position_scale,
        body_raw_to_flu=body_raw_to_flu,
        body_origin_raw=body_origin_raw,
    )
    quat = _convert_quat_xyzw(
        pose[3:7],
        mode,
        orientation_mode,
        body_raw_to_flu=body_raw_to_flu,
    )
    return {
        "position": {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])},
        "quaternion": {
            "x": float(quat[0]),
            "y": float(quat[1]),
            "z": float(quat[2]),
            "w": float(quat[3]),
        },
    }


def _position_to_vr_json(
    pos: np.ndarray,
    mode: PicoCoordinateMode,
    position_scale: float,
    body_raw_to_flu: np.ndarray | None = None,
    body_origin_raw: np.ndarray | None = None,
) -> dict[str, float]:
    p = _convert_position(
        pos,
        mode,
        position_scale,
        body_raw_to_flu=body_raw_to_flu,
        body_origin_raw=body_origin_raw,
    )
    return {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}


def _get_body_joints(xrt: Any) -> np.ndarray:
    joints = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64)
    if joints.shape[0] < 24 or joints.shape[1] < 7:
        raise ValueError(f"expected body joints shape at least (24, 7), got {joints.shape}")
    return joints


def _get_timestamp_ns(xrt: Any) -> int:
    # GR00T's XRoboToolkit integration keys freshness from get_time_stamp_ns().
    # On some setups get_body_timestamp_ns() can stay constant even while poses
    # continue to stream, which would make the bridge falsely mark tracking stale.
    for name in ("get_time_stamp_ns", "get_body_timestamp_ns"):
        try:
            return int(getattr(xrt, name)())
        except Exception:
            continue
    return time.time_ns()


def _get_controller_pose(xrt: Any, side: Literal["left", "right"]) -> np.ndarray:
    name = f"get_{side}_controller_pose"
    return _pose_array(getattr(xrt, name)(), f"{side} controller")


def _get_body_pose(
    joints: np.ndarray, side: Literal["left", "right"], source: Literal["wrist", "hand"]
) -> np.ndarray:
    if side == "left":
        idx = _BODY_LEFT_WRIST if source == "wrist" else _BODY_LEFT_HAND
    else:
        idx = _BODY_RIGHT_WRIST if source == "wrist" else _BODY_RIGHT_HAND
    return _pose_array(joints[idx], f"{side} body {source}")


def _synthetic_elbow(
    joints: np.ndarray, side: Literal["left", "right"], hand_pos: np.ndarray
) -> np.ndarray:
    """Estimate an elbow from shoulder and hand/controller position in raw XRT space."""
    left_shoulder = joints[_BODY_LEFT_SHOULDER, :3]
    right_shoulder = joints[_BODY_RIGHT_SHOULDER, :3]
    shoulder = left_shoulder if side == "left" else right_shoulder
    neck = joints[_BODY_NECK, :3]
    pelvis = joints[_BODY_PELVIS, :3]

    body_left = _normalize_vec(left_shoulder - right_shoulder, "synthetic body left")
    body_up = _normalize_vec(neck - pelvis, "synthetic body up")
    outward = body_left if side == "left" else -body_left

    shoulder_to_hand = hand_pos - shoulder
    reach = float(np.linalg.norm(shoulder_to_hand))
    if reach <= 1e-6:
        return shoulder.copy()

    # Keep the elbow moving with the commanded hand while biasing it slightly
    # outward/downward so the IK has a human-like arm plane instead of a straight
    # shoulder->hand line.
    outward_offset = min(0.12, max(0.04, 0.22 * reach))
    downward_offset = min(0.08, max(0.02, 0.12 * reach))
    return shoulder + 0.55 * shoulder_to_hand + outward_offset * outward - downward_offset * body_up


class PicoAxolBridge:
    """Read XRoboToolkit state and send Axol-compatible VRFrame JSON."""

    def __init__(self, config: PicoBridgeConfig) -> None:
        self.config = config
        self.state = _AxolState.TELEOP
        self._pending_recording_at: float | None = None
        self._prev_a = False
        self._prev_b = False
        self._prev_x = False
        self._prev_y = False
        self._seq = 0
        self._latest_body_joints: np.ndarray | None = None
        self._latest_body_timestamp_ns: int | None = None
        self._latest_body_seen_monotonic: float | None = None
        self._last_report = 0.0
        self._sent = 0
        self._service_proc: subprocess.Popen[Any] | None = None
        self._last_debug: dict[str, Any] = {}
        self._frozen_l_elbow: np.ndarray | None = None
        self._frozen_r_elbow: np.ndarray | None = None
        self._body_raw_to_flu: np.ndarray | None = None
        self._body_origin_raw: np.ndarray | None = None

    def _uri(self) -> str:
        scheme = "wss" if self.config.tls else "ws"
        path = self.config.path if self.config.path.startswith("/") else f"/{self.config.path}"
        return f"{scheme}://{self.config.host}:{self.config.port}{path}"

    def _ssl_context(self) -> ssl.SSLContext | None:
        if not self.config.tls:
            return None
        if self.config.verify_tls:
            return ssl.create_default_context()
        return ssl._create_unverified_context()

    async def run(self) -> None:
        xrt = _load_xrt()
        self._service_proc = _maybe_start_service(self.config)
        try:
            xrt.init()
            await self._wait_for_body_tracking(xrt)
            if self.config.dry_run:
                await self._dry_run_loop(xrt)
                return
            await self._websocket_loop(xrt)
        finally:
            try:
                xrt.close()
            except Exception:
                pass
            if self._service_proc is not None and self._service_proc.poll() is None:
                self._service_proc.terminate()

    async def _wait_for_body_tracking(self, xrt: Any) -> None:
        _logger.info("waiting for XRoboToolkit body tracking...")
        deadline = (
            time.monotonic() + self.config.wait_body_timeout_s
            if self.config.wait_body_timeout_s > 0
            else None
        )
        while True:
            try:
                if xrt.is_body_data_available():
                    joints = _get_body_joints(xrt)
                    self._latest_body_joints = joints
                    self._latest_body_timestamp_ns = _get_timestamp_ns(xrt)
                    self._latest_body_seen_monotonic = time.monotonic()
                    _logger.info("body tracking ready")
                    return
            except Exception as exc:
                _logger.debug("body tracking not ready: %s", exc)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "XRoboToolkit body tracking did not become available. "
                    "Check PICO app: Head + Controller + Send, and Full body tracking."
                )
            await asyncio.sleep(0.1)

    async def _websocket_loop(self, xrt: Any) -> None:
        import websockets

        uri = self._uri()
        _logger.info("connecting to Axol VR server: %s", uri)
        async with websockets.connect(uri, ssl=self._ssl_context()) as ws:
            receiver = asyncio.create_task(self._receive_feedback(ws))
            try:
                period = 1.0 / max(1.0, self.config.frequency)
                while True:
                    t0 = time.monotonic()
                    frame = self._build_frame(xrt)
                    await ws.send(json.dumps(frame, separators=(",", ":")))
                    self._sent += 1
                    self._report_if_due()
                    await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))
            except _ExitBridge:
                _logger.info("exit requested from PICO controller")
            finally:
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass

    async def _receive_feedback(self, ws: Any) -> None:
        async for text in ws:
            try:
                msg = json.loads(text)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "state":
                value = msg.get("value")
                if isinstance(value, str):
                    self.state = value
                    if value == _AxolState.SAVING:
                        self._pending_recording_at = None
                    _logger.info("server state -> %s", value)

    async def _dry_run_loop(self, xrt: Any) -> None:
        period = 1.0 / max(1.0, self.config.frequency)
        while True:
            t0 = time.monotonic()
            frame = self._build_frame(xrt)
            self._sent += 1
            self._report_if_due(extra=json.dumps(frame)[:300])
            await asyncio.sleep(max(0.0, period - (time.monotonic() - t0)))

    def _update_body_sample(self, xrt: Any) -> None:
        if not xrt.is_body_data_available():
            return
        stamp_ns = _get_timestamp_ns(xrt)
        if self.config.freshness_mode == "local":
            self._latest_body_joints = _get_body_joints(xrt)
            self._latest_body_timestamp_ns = stamp_ns
            self._latest_body_seen_monotonic = time.monotonic()
            return
        if (
            self._latest_body_timestamp_ns is not None
            and stamp_ns == self._latest_body_timestamp_ns
        ):
            return
        self._latest_body_joints = _get_body_joints(xrt)
        self._latest_body_timestamp_ns = stamp_ns
        self._latest_body_seen_monotonic = time.monotonic()

    def _body_is_fresh(self) -> bool:
        if self._latest_body_seen_monotonic is None:
            return False
        return (time.monotonic() - self._latest_body_seen_monotonic) <= self.config.stale_timeout_s

    def _ensure_body_calibration(self, joints: np.ndarray) -> None:
        if self.config.coordinate_mode != "body":
            return
        if self._body_raw_to_flu is not None and self._body_origin_raw is not None:
            return

        pelvis = joints[_BODY_PELVIS, :3]
        neck = joints[_BODY_NECK, :3]
        left_shoulder = joints[_BODY_LEFT_SHOULDER, :3]
        right_shoulder = joints[_BODY_RIGHT_SHOULDER, :3]

        left = _normalize_vec(left_shoulder - right_shoulder, "body left")
        up = _normalize_vec(neck - pelvis, "body up")
        up = _normalize_vec(up - left * float(np.dot(up, left)), "body up projected")
        forward = _normalize_vec(np.cross(left, up), "body forward")
        if self.config.body_forward_sign < 0:
            forward = -forward
        left = _normalize_vec(np.cross(up, forward), "body left orthogonalized")
        up = _normalize_vec(np.cross(forward, left), "body up orthogonalized")

        self._body_raw_to_flu = np.stack([forward, left, up], axis=0)
        self._body_origin_raw = pelvis.astype(np.float64).copy()
        _logger.info(
            "body calibration: forward=%s left=%s up=%s origin=%s",
            np.round(forward, 4).tolist(),
            np.round(left, 4).tolist(),
            np.round(up, 4).tolist(),
            np.round(self._body_origin_raw, 4).tolist(),
        )

    def _build_frame(self, xrt: Any) -> dict[str, Any]:
        self._update_body_sample(xrt)
        joints = self._latest_body_joints
        if joints is None:
            raise RuntimeError("no body joints available")
        self._ensure_body_calibration(joints)

        fresh = self._body_is_fresh()
        l_controller_pose: np.ndarray | None = None
        r_controller_pose: np.ndarray | None = None
        if self.config.ee_source == "controller":
            l_controller_pose = _get_controller_pose(xrt, "left")
            r_controller_pose = _get_controller_pose(xrt, "right")
            l_ee_pose = l_controller_pose
            r_ee_pose = r_controller_pose
        else:
            l_ee_pose = _get_body_pose(joints, "left", self.config.ee_source)
            r_ee_pose = _get_body_pose(joints, "right", self.config.ee_source)
            if self.config.orientation_mode == "controller":
                l_controller_pose = _get_controller_pose(xrt, "left")
                r_controller_pose = _get_controller_pose(xrt, "right")
                l_ee_pose = l_ee_pose.copy()
                r_ee_pose = r_ee_pose.copy()
                l_ee_pose[3:7] = l_controller_pose[3:7]
                r_ee_pose[3:7] = r_controller_pose[3:7]

        if self.config.elbow_source == "synthetic":
            l_elbow = _synthetic_elbow(joints, "left", l_ee_pose[:3])
            r_elbow = _synthetic_elbow(joints, "right", r_ee_pose[:3])
        else:
            l_elbow = joints[_BODY_LEFT_ELBOW, :3]
            r_elbow = joints[_BODY_RIGHT_ELBOW, :3]
        if self.config.elbow_source == "frozen":
            if self._frozen_l_elbow is None:
                self._frozen_l_elbow = l_elbow.copy()
                self._frozen_r_elbow = r_elbow.copy()
            l_elbow = self._frozen_l_elbow
            r_elbow = self._frozen_r_elbow

        left_trigger = _apply_deadzone(
            _analog(xrt, "get_left_trigger"), self.config.trigger_deadzone
        )
        right_trigger = _apply_deadzone(
            _analog(xrt, "get_right_trigger"), self.config.trigger_deadzone
        )
        left_grip = _apply_deadzone(_analog(xrt, "get_left_grip"), self.config.grip_deadzone)
        right_grip = _apply_deadzone(_analog(xrt, "get_right_grip"), self.config.grip_deadzone)

        a_pressed = _button(xrt, "get_A_button")
        b_pressed = _button(xrt, "get_B_button")
        x_pressed = _button(xrt, "get_X_button")
        y_pressed = _button(xrt, "get_Y_button")

        a_edge = a_pressed and not self._prev_a
        b_edge = b_pressed and not self._prev_b
        x_edge = x_pressed and not self._prev_x
        y_edge = y_pressed and not self._prev_y
        self._prev_a = a_pressed
        self._prev_b = b_pressed
        self._prev_x = x_pressed
        self._prev_y = y_pressed

        if y_edge:
            raise _ExitBridge()

        reset = False
        saving = self.state == _AxolState.SAVING
        if x_edge and not saving:
            reset = True
            if self.state == _AxolState.RECORDING or self._pending_recording_at is not None:
                self.state = _AxolState.DATA_COLLECTION
                self._pending_recording_at = None

        if b_edge and self.state not in (_AxolState.RECORDING, _AxolState.SAVING):
            if self._pending_recording_at is None:
                self.state = (
                    _AxolState.DATA_COLLECTION
                    if self.state == _AxolState.TELEOP
                    else _AxolState.TELEOP
                )

        if a_edge and not saving:
            if self.state == _AxolState.RECORDING:
                self.state = _AxolState.DATA_COLLECTION
            elif self.state == _AxolState.DATA_COLLECTION and self._pending_recording_at is None:
                self._pending_recording_at = time.monotonic()
            elif self._pending_recording_at is not None:
                self._pending_recording_at = None

        if (
            self._pending_recording_at is not None
            and time.monotonic() - self._pending_recording_at >= self.config.record_countdown_s
        ):
            self.state = _AxolState.RECORDING
            self._pending_recording_at = None

        locks_allowed = fresh and self.state not in (_AxolState.SAVING, _AxolState.ERROR)
        if self.config.auto_engage:
            l_lock = locks_allowed
            r_lock = locks_allowed
        else:
            l_lock = locks_allowed and left_grip >= self.config.lock_threshold
            r_lock = locks_allowed and right_grip >= self.config.lock_threshold

        self._seq += 1
        l_vr = _pose_to_vr_json(
            l_ee_pose,
            self.config.coordinate_mode,
            self.config.orientation_mode,
            self.config.position_scale,
            body_raw_to_flu=self._body_raw_to_flu,
            body_origin_raw=self._body_origin_raw,
        )
        r_vr = _pose_to_vr_json(
            r_ee_pose,
            self.config.coordinate_mode,
            self.config.orientation_mode,
            self.config.position_scale,
            body_raw_to_flu=self._body_raw_to_flu,
            body_origin_raw=self._body_origin_raw,
        )
        l_elbow_vr = _position_to_vr_json(
            l_elbow,
            self.config.coordinate_mode,
            self.config.position_scale,
            body_raw_to_flu=self._body_raw_to_flu,
            body_origin_raw=self._body_origin_raw,
        )
        r_elbow_vr = _position_to_vr_json(
            r_elbow,
            self.config.coordinate_mode,
            self.config.position_scale,
            body_raw_to_flu=self._body_raw_to_flu,
            body_origin_raw=self._body_origin_raw,
        )
        self._last_debug = {
            "fresh": fresh,
            "locks": (bool(l_lock), bool(r_lock)),
            "inputs": (
                float(left_grip),
                float(right_grip),
                float(left_trigger),
                float(right_trigger),
            ),
            "buttons": (a_pressed, b_pressed, x_pressed, y_pressed),
            "l_ee": l_vr["position"],
            "r_ee": r_vr["position"],
            "l_elbow": l_elbow_vr,
            "r_elbow": r_elbow_vr,
        }
        return {
            "l_ee": l_vr,
            "r_ee": r_vr,
            "l_elbow": l_elbow_vr,
            "r_elbow": r_elbow_vr,
            "l_grip": _gripper_value(left_trigger, self.config.gripper_mode),
            "r_grip": _gripper_value(right_trigger, self.config.gripper_mode),
            "l_lock": bool(l_lock),
            "r_lock": bool(r_lock),
            "reset": bool(reset),
            "state": self.state,
            "seq": self._seq,
        }

    def _report_if_due(self, extra: str | None = None) -> None:
        now = time.monotonic()
        if now - self._last_report < self.config.log_every_s:
            return
        body_age = (
            now - self._latest_body_seen_monotonic
            if self._latest_body_seen_monotonic is not None
            else float("inf")
        )
        _logger.info(
            "pico bridge: sent=%d state=%s body_age=%.0fms ee_source=%s "
            "coord=%s orient=%s scale=%.2f fresh=%s gripper=%s "
            "elbow=%s locks=%s inputs=%s buttons=%s%s",
            self._sent,
            self.state,
            body_age * 1e3,
            self.config.ee_source,
            self.config.coordinate_mode,
            self.config.orientation_mode,
            self.config.position_scale,
            self.config.freshness_mode,
            self.config.gripper_mode,
            self.config.elbow_source,
            self._last_debug.get("locks"),
            self._last_debug.get("inputs"),
            self._last_debug.get("buttons"),
            f" sample={extra}" if extra else "",
        )
        if _logger.isEnabledFor(logging.DEBUG) and self._last_debug:
            _logger.debug(
                "pico poses: l_ee=%s r_ee=%s l_elbow=%s r_elbow=%s",
                self._last_debug.get("l_ee"),
                self._last_debug.get("r_ee"),
                self._last_debug.get("l_elbow"),
                self._last_debug.get("r_elbow"),
            )
        self._last_report = now


class _ExitBridge(Exception):
    pass
