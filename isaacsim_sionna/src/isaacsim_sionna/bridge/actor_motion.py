"""Config-driven actor motion utilities for Isaac-Sionna pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


def _as_vec3(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return [float(v) for v in default]
    return [float(value[0]), float(value[1]), float(value[2])]


def _as_quat_wxyz(value: Any, default: list[float]) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return [float(v) for v in default]
    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]


def _quat_normalize(q: list[float]) -> list[float]:
    n = math.sqrt(sum(v * v for v in q))
    if n <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n]


def _quat_conjugate(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def yaw_to_quat_wxyz(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def angular_velocity_from_quats(prev_q: list[float], curr_q: list[float], dt: float) -> list[float]:
    if dt <= 1e-9:
        return [0.0, 0.0, 0.0]
    q0 = _quat_normalize(prev_q)
    q1 = _quat_normalize(curr_q)
    dq = _quat_multiply(q1, _quat_conjugate(q0))
    if dq[0] < 0.0:
        dq = [-dq[0], -dq[1], -dq[2], -dq[3]]

    w = max(-1.0, min(1.0, dq[0]))
    v_norm = math.sqrt(max(0.0, dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3]))
    if v_norm <= 1e-12:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(v_norm, w)
    axis = [dq[1] / v_norm, dq[2] / v_norm, dq[3] / v_norm]
    scale = angle / dt
    return [axis[0] * scale, axis[1] * scale, axis[2] * scale]


@dataclass
class MotionPose:
    prim_path: str
    position_xyz: list[float]
    orientation_quat_wxyz: list[float]
    motion_type: str


class BaseMotionController:
    def __init__(self, prim_path: str, motion_type: str):
        self.prim_path = prim_path
        self.motion_type = motion_type

    def pose_at(self, timestamp_sim: float, frame_idx: int) -> MotionPose | None:
        raise NotImplementedError


class ManualMotionController(BaseMotionController):
    def __init__(self, prim_path: str, cfg: dict[str, Any]):
        super().__init__(prim_path=prim_path, motion_type="manual")
        manual_cfg = cfg.get("manual", {})
        fixed = manual_cfg.get("fixed_pose", {})
        self.position_xyz = _as_vec3(fixed.get("pos_xyz"), [0.0, 0.0, 0.0])
        self.orientation_quat_wxyz = _as_quat_wxyz(fixed.get("quat_wxyz"), [1.0, 0.0, 0.0, 0.0])

    def pose_at(self, timestamp_sim: float, frame_idx: int) -> MotionPose | None:
        _ = timestamp_sim, frame_idx
        return MotionPose(
            prim_path=self.prim_path,
            position_xyz=list(self.position_xyz),
            orientation_quat_wxyz=list(self.orientation_quat_wxyz),
            motion_type=self.motion_type,
        )


class TrajectoryMotionController(BaseMotionController):
    def __init__(self, prim_path: str, cfg: dict[str, Any]):
        super().__init__(prim_path=prim_path, motion_type="trajectory")
        traj = cfg.get("trajectory", {})
        self.kind = str(traj.get("kind", "waypoints")).lower()
        self.loop = bool(traj.get("loop", True))
        self.speed_mps = float(traj.get("speed_mps", 1.0))
        self.face_forward = bool(traj.get("face_forward", True))

        self.waypoints_xyz = [[float(x), float(y), float(z)] for x, y, z in traj.get("waypoints_xyz", [])]
        self.circle_center_xyz = _as_vec3(traj.get("center_xyz"), [0.0, 0.0, 0.0])
        self.circle_radius_m = float(traj.get("radius_m", 1.0))
        self.circle_angular_speed_radps = float(traj.get("angular_speed_radps", 0.5))

        self._segments: list[tuple[list[float], list[float], float]] = []
        self._length_total = 0.0
        if self.kind == "waypoints" and len(self.waypoints_xyz) >= 2:
            pts = list(self.waypoints_xyz)
            if self.loop and pts[0] != pts[-1]:
                pts.append(list(pts[0]))
            for a, b in zip(pts[:-1], pts[1:]):
                d = math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2 + (b[2] - a[2]) ** 2)
                if d > 1e-9:
                    self._segments.append((a, b, d))
                    self._length_total += d

    def _waypoint_pose(self, timestamp_sim: float) -> tuple[list[float], list[float]]:
        if not self._segments or self._length_total <= 1e-9:
            p = self.waypoints_xyz[0] if self.waypoints_xyz else [0.0, 0.0, 0.0]
            return list(p), [1.0, 0.0, 0.0, 0.0]

        s = self.speed_mps * max(0.0, float(timestamp_sim))
        if self.loop:
            s = s % self._length_total
        else:
            s = min(s, self._length_total)

        traveled = 0.0
        tangent = [1.0, 0.0, 0.0]
        for a, b, d in self._segments:
            if traveled + d >= s:
                u = (s - traveled) / d if d > 0 else 0.0
                pos = [a[i] + (b[i] - a[i]) * u for i in range(3)]
                tangent = [(b[i] - a[i]) / d for i in range(3)] if d > 0 else tangent
                break
            traveled += d
        else:
            a, b, d = self._segments[-1]
            pos = list(b)
            tangent = [(b[i] - a[i]) / d for i in range(3)] if d > 0 else tangent

        if self.face_forward:
            yaw = math.atan2(tangent[1], tangent[0])
            quat = yaw_to_quat_wxyz(yaw)
        else:
            quat = [1.0, 0.0, 0.0, 0.0]
        return pos, quat

    def _circle_pose(self, timestamp_sim: float) -> tuple[list[float], list[float]]:
        theta = self.circle_angular_speed_radps * max(0.0, float(timestamp_sim))
        cx, cy, cz = self.circle_center_xyz
        r = max(0.0, self.circle_radius_m)
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        z = cz
        yaw = theta + 0.5 * math.pi if self.face_forward else 0.0
        quat = yaw_to_quat_wxyz(yaw)
        return [x, y, z], quat

    def pose_at(self, timestamp_sim: float, frame_idx: int) -> MotionPose | None:
        _ = frame_idx
        if self.kind == "circle":
            pos, quat = self._circle_pose(timestamp_sim)
        else:
            pos, quat = self._waypoint_pose(timestamp_sim)
        return MotionPose(
            prim_path=self.prim_path,
            position_xyz=pos,
            orientation_quat_wxyz=quat,
            motion_type=self.motion_type,
        )


class AnimationMotionController(BaseMotionController):
    """Animation mode is passive: USD animation drives the prim."""

    def __init__(self, prim_path: str, cfg: dict[str, Any]):
        super().__init__(prim_path=prim_path, motion_type="animation")
        anim = cfg.get("animation", {})
        self.clip = str(anim.get("clip", ""))
        self.playback_speed = float(anim.get("playback_speed", 1.0))

    def pose_at(self, timestamp_sim: float, frame_idx: int) -> MotionPose | None:
        _ = timestamp_sim, frame_idx
        return None


class ActorMotionManager:
    def __init__(self, cfg: dict[str, Any] | None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.update_every_tick = bool(cfg.get("update_every_tick", True))
        self._controllers: dict[str, BaseMotionController] = {}

        actors_cfg = cfg.get("actors", [])
        if not isinstance(actors_cfg, list):
            actors_cfg = []

        for item in actors_cfg:
            if not isinstance(item, dict):
                continue
            prim_path = str(item.get("prim_path", "")).strip()
            if not prim_path:
                continue
            motion_type = str(item.get("motion_type", "manual")).lower()
            if motion_type == "trajectory":
                controller = TrajectoryMotionController(prim_path=prim_path, cfg=item)
            elif motion_type == "animation":
                controller = AnimationMotionController(prim_path=prim_path, cfg=item)
            else:
                controller = ManualMotionController(prim_path=prim_path, cfg=item)
            self._controllers[prim_path] = controller

    def configured_actor_paths(self) -> list[str]:
        return list(self._controllers.keys())

    def motion_type_for(self, prim_path: str) -> str:
        ctrl = self._controllers.get(prim_path)
        if ctrl is None:
            return "none"
        return ctrl.motion_type

    def target_poses(self, timestamp_sim: float, frame_idx: int) -> list[MotionPose]:
        if not self.enabled:
            return []
        out: list[MotionPose] = []
        for ctrl in self._controllers.values():
            pose = ctrl.pose_at(timestamp_sim=timestamp_sim, frame_idx=frame_idx)
            if pose is not None:
                out.append(pose)
        return out
