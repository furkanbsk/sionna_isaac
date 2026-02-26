"""Isaac Sim RGB camera adapter for synchronized frame capture."""

from __future__ import annotations

import logging
from pathlib import Path
import time
from typing import Any
import math

import numpy as np

logger = logging.getLogger(__name__)


class CameraAdapter:
    """Optional RGB frame capture via Isaac Sim Replicator."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        isaac_cfg = config.get("isaac", {}) if isinstance(config, dict) else {}
        camera_cfg = isaac_cfg.get("camera", {}) if isinstance(isaac_cfg, dict) else {}
        project_cfg = config.get("project", {}) if isinstance(config, dict) else {}

        self.enabled = bool(camera_cfg.get("enabled", False))
        self.backend = str(camera_cfg.get("backend", "replicator")).strip().lower()
        if self.backend not in {"replicator"}:
            self.backend = "replicator"
        self.position_xyz = [float(v) for v in camera_cfg.get("position_xyz", [0.0, 0.0, 10.0])]
        self.target_xyz = [float(v) for v in camera_cfg.get("target_xyz", [0.0, 0.0, 0.0])]
        orientation_rpy = camera_cfg.get("orientation_rpy_deg")
        if isinstance(orientation_rpy, (list, tuple)) and len(orientation_rpy) == 3:
            self.orientation_rpy_deg = [float(v) for v in orientation_rpy]
        else:
            self.orientation_rpy_deg = None
        orientation_quat = camera_cfg.get("orientation_quat_wxyz")
        if isinstance(orientation_quat, (list, tuple)) and len(orientation_quat) == 4:
            self.orientation_quat_wxyz = [float(v) for v in orientation_quat]
        else:
            self.orientation_quat_wxyz = None
        resolution = camera_cfg.get("resolution", [1280, 720])
        self.width = int(resolution[0]) if isinstance(resolution, (list, tuple)) and len(resolution) >= 2 else 1280
        self.height = int(resolution[1]) if isinstance(resolution, (list, tuple)) and len(resolution) >= 2 else 720
        self.image_format = str(camera_cfg.get("format", "png")).lower()
        if self.image_format not in {"png", "jpg", "jpeg"}:
            self.image_format = "png"
        self.fov_deg = float(camera_cfg.get("fov_deg", 70.0))
        self.clip_near_m = float(camera_cfg.get("clip_near_m", 0.05))
        self.clip_far_m = float(camera_cfg.get("clip_far_m", 500.0))
        self.look_at_mode = str(camera_cfg.get("look_at_mode", "target")).strip().lower()
        if self.look_at_mode not in {"target", "orientation"}:
            self.look_at_mode = "target"
        self.output_subdir = str(camera_cfg.get("output_subdir", "renders"))
        self.camera_name = str(camera_cfg.get("name", "camera_main"))
        self.prim_path = str(camera_cfg.get("prim_path", f"/World/{self.camera_name}"))
        self.warmup_steps = max(0, int(camera_cfg.get("warmup_steps", 1)))
        self.capture_max_attempts = max(1, int(camera_cfg.get("capture_max_attempts", 2)))
        self.capture_timeout_s = float(camera_cfg.get("capture_timeout_s", 1.0))
        self.normalize_intensity = bool(camera_cfg.get("normalize_intensity", True))
        self.normalize_p_low = float(camera_cfg.get("normalize_p_low", 1.0))
        self.normalize_p_high = float(camera_cfg.get("normalize_p_high", 99.0))
        self.output_root = Path(project_cfg.get("output_root", "isaacsim_sionna/data/raw"))

        self._rep = None
        self._camera = None
        self._render_product = None
        self._rgb_annotator = None
        self._renders_dir: Path | None = None
        self._initialized = False
        self._init_error: str | None = None
        self._warned_keys: set[str] = set()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message)

    @property
    def init_error(self) -> str | None:
        return self._init_error

    def initialize(self) -> bool:
        """Create camera render product and attach RGB annotator."""
        if not self.enabled or self._initialized:
            return self._initialized

        try:
            import omni.replicator.core as rep  # pylint: disable=import-outside-toplevel
        except Exception as exc:
            self._init_error = f"replicator_import_failed:{exc}"
            self._warn_once("init_import", f"[CameraAdapter] WARN: failed to import Replicator: {exc}")
            return False

        try:
            self._rep = rep
            self._renders_dir = self.output_root / self.output_subdir
            self._renders_dir.mkdir(parents=True, exist_ok=True)

            # Replicator does not consistently expose prim_path pinning across versions.
            # We use deterministic name + pose and keep prim_path in metadata/config.
            cam_kwargs = {"position": tuple(self.position_xyz)}
            # Optional camera intrinsics that may differ across Isaac versions.
            # We use progressive fallback if a key is unsupported.
            cam_kwargs["fov"] = float(self.fov_deg)
            cam_kwargs["clipping_range"] = (float(self.clip_near_m), float(self.clip_far_m))
            if self.look_at_mode == "orientation" and self.orientation_rpy_deg is not None:
                cam_kwargs["rotation"] = tuple(self.orientation_rpy_deg)
            elif self.look_at_mode == "orientation" and self.orientation_quat_wxyz is not None:
                cam_kwargs["rotation"] = tuple(self._quat_wxyz_to_euler_deg(self.orientation_quat_wxyz))
            else:
                cam_kwargs["look_at"] = tuple(self.target_xyz)
            self._camera = self._create_camera_with_fallback(rep, cam_kwargs)
            self._render_product = rep.create.render_product(self._camera, (self.width, self.height))
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([self._render_product])

            for _ in range(self.warmup_steps):
                rep.orchestrator.step()

            self._initialized = True
            self._init_error = None
            return True
        except Exception as exc:
            self._init_error = f"camera_init_failed:{exc}"
            self._warn_once("init_runtime", f"[CameraAdapter] WARN: camera initialization failed: {exc}")
            self.close()
            return False

    @staticmethod
    def _create_camera_with_fallback(rep: Any, cam_kwargs: dict[str, Any]) -> Any:
        """Create a camera while gracefully dropping unsupported kwargs."""
        kwargs = dict(cam_kwargs)
        drop_order = ["clipping_range", "fov"]
        for _ in range(len(drop_order) + 1):
            try:
                return rep.create.camera(**kwargs)
            except TypeError:
                if not drop_order:
                    raise
                key = drop_order.pop(0)
                if key in kwargs:
                    kwargs.pop(key)
                # If key wasn't present, continue to next in drop_order
        raise TypeError("Failed to create camera after exhausting fallback kwargs")

    @staticmethod
    def _quat_wxyz_to_euler_deg(quat_wxyz: list[float]) -> list[float]:
        """Convert quaternion [w,x,y,z] to Euler XYZ in degrees."""
        w, x, y, z = [float(v) for v in quat_wxyz]

        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)

        return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]

    @staticmethod
    def _save_image(path: Path, rgb: np.ndarray, image_format: str) -> None:
        arr = np.asarray(rgb)
        if arr.ndim != 3:
            raise ValueError(f"Expected RGB image rank 3, got shape={arr.shape}")
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            img = Image.fromarray(arr, mode="RGB")
            if image_format in {"jpg", "jpeg"}:
                img.save(str(path), format="JPEG", quality=95)
            else:
                img.save(str(path), format="PNG")
            return
        except Exception:
            pass

        try:
            import imageio.v2 as imageio  # pylint: disable=import-outside-toplevel

            imageio.imwrite(str(path), arr)
            return
        except Exception as exc:
            raise RuntimeError("No image writer backend available (PIL/imageio)") from exc

    @staticmethod
    def _normalize_rgb(rgb: np.ndarray, p_low: float, p_high: float) -> np.ndarray:
        arr = np.asarray(rgb, dtype=np.float32)
        if arr.ndim != 3:
            return np.asarray(rgb)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]

        # Robust percentile stretch to reduce washout in high-dynamic-range scenes.
        q_lo = float(np.clip(p_low, 0.0, 99.999))
        q_hi = float(np.clip(p_high, q_lo, 100.0))
        lo = float(np.percentile(arr, q_lo))
        hi = float(np.percentile(arr, q_hi))
        if hi - lo <= 1e-6:
            return np.clip(arr, 0.0, 255.0).astype(np.uint8)
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        return arr.astype(np.uint8)

    def capture(self, frame_idx: int) -> dict[str, Any] | None:
        """Capture one RGB frame and return index metadata."""
        if not self.enabled:
            return None
        if not self._initialized:
            ok = self.initialize()
            if not ok:
                return None
        if self._rep is None or self._rgb_annotator is None or self._renders_dir is None:
            return None

        # Flush render frames with bounded attempts/timeout to avoid headless stalls.
        start_t = time.perf_counter()
        rgb = None
        for attempt_idx in range(self.capture_max_attempts):
            if (time.perf_counter() - start_t) > self.capture_timeout_s:
                self._warn_once(
                    "capture_timeout",
                    (
                        "[CameraAdapter] WARN: capture timed out "
                        f"after {self.capture_timeout_s:.3f}s and {attempt_idx} attempts"
                    ),
                )
                return None
            try:
                self._rep.orchestrator.step()
                rgb = self._rgb_annotator.get_data()
            except Exception as exc:
                self._warn_once("capture", f"[CameraAdapter] WARN: capture failed: {exc}")
                return None
            if rgb is not None:
                break
        if rgb is None:
            self._warn_once(
                "capture_empty",
                f"[CameraAdapter] WARN: capture returned no RGB data after {self.capture_max_attempts} attempts",
            )
            return None

        ext = "jpg" if self.image_format in {"jpg", "jpeg"} else "png"
        rel = Path(self.output_subdir) / f"frame_{int(frame_idx):04d}.{ext}"
        out_path = self.output_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb_np = np.asarray(rgb)
        if self.normalize_intensity:
            rgb_np = self._normalize_rgb(rgb_np, self.normalize_p_low, self.normalize_p_high)
        self._save_image(out_path, rgb_np, image_format=ext)

        return {
            "file": str(rel),
            "camera_name": self.camera_name,
            "camera_prim_path": self.prim_path,
            "width": int(self.width),
            "height": int(self.height),
        }

    def close(self) -> None:
        if self._rgb_annotator is not None:
            if self._render_product is not None:
                try:
                    self._rgb_annotator.detach([self._render_product])
                except Exception:
                    pass
            else:
                try:
                    self._rgb_annotator.detach([])
                except Exception:
                    pass
        self._rgb_annotator = None
        self._render_product = None
        self._camera = None
        self._rep = None
        self._initialized = False
