"""Isaac Sim adapter.

This module owns direct calls to Isaac Sim APIs and exposes a minimal
pipeline-friendly interface:
1. start
2. step
3. get_state
4. stop
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from isaacsim_sionna.bridge.actor_motion import ActorMotionManager, angular_velocity_from_quats
from isaacsim_sionna.bridge.camera_adapter import CameraAdapter
from isaacsim_sionna.bridge.usd_to_sionna import compute_global_bbox, extract_mesh_aabbs_from_usd_file
from isaacsim_sionna.utils.reproducibility import seed_isaac_runtime


class IsaacAdapter:
    """Adapter for stepping Isaac Sim and exporting pose state."""

    def __init__(self, config: dict):
        self.config = config

        runtime_cfg = config.get("runtime", {})
        isaac_cfg = config.get("isaac", {})
        scenario_cfg = config.get("scenario", {})
        prim_cfg = isaac_cfg.get("prim_paths", {})
        anchor_cfg = isaac_cfg.get("anchors", {})
        geometry_cfg = isaac_cfg.get("geometry", {})
        mesh_cfg = geometry_cfg.get("mesh", {})
        camera_cfg = isaac_cfg.get("camera", {})
        actor_motion_cfg = isaac_cfg.get("actor_motion", {})
        project_cfg = config.get("project", {})

        self.isaac_fps = float(runtime_cfg.get("isaac_fps", 60.0))
        self.headless = bool(isaac_cfg.get("headless", True))
        self.render = bool(isaac_cfg.get("render", False))
        self.scene_usd = scenario_cfg.get("scene_usd")

        self.tx_prim_path = prim_cfg.get("tx", "/World/tx")
        self.rx_prim_path = prim_cfg.get("rx", "/World/rx")
        self.actor_prim_paths = list(prim_cfg.get("actors", ["/World/actor_0"]))
        self._actor_motion = ActorMotionManager(actor_motion_cfg)
        for prim_path in self._actor_motion.configured_actor_paths():
            if prim_path not in self.actor_prim_paths:
                self.actor_prim_paths.append(prim_path)

        self.auto_anchor_from_bbox = bool(anchor_cfg.get("auto_from_bbox", True))
        self.anchor_height_m = float(anchor_cfg.get("height_m", 1.5))
        self.tx_offset_norm = [float(v) for v in anchor_cfg.get("tx_offset_norm", [0.2, 0.2])]
        self.rx_offset_norm = [float(v) for v in anchor_cfg.get("rx_offset_norm", [0.8, 0.8])]
        self.manual_tx_xyz = anchor_cfg.get("manual_tx_xyz")
        self.manual_rx_xyz = anchor_cfg.get("manual_rx_xyz")
        self.max_meshes = int(anchor_cfg.get("max_meshes", 256))

        self.geometry_mode = str(geometry_cfg.get("mode", "aabb")).lower()
        self.geometry_mesh_format = str(mesh_cfg.get("format", "ply")).lower()
        self.geometry_mesh_output_dir = str(
            mesh_cfg.get("output_dir", "isaacsim_sionna/data/scenes_sionna/runtime_meshes")
        )
        self.geometry_mesh_include_regex = mesh_cfg.get("include_regex")
        self.geometry_mesh_exclude_regex = mesh_cfg.get("exclude_regex")
        self.geometry_mesh_max_meshes = int(mesh_cfg.get("max_meshes", 256))
        self.camera_enabled = bool(camera_cfg.get("enabled", False))
        self.seed = int(project_cfg.get("seed", 42))

        self._simulation_app = None
        self._world = None
        self._World = None
        self._stage_source = "autogen"
        self._frame_idx = 0
        self._missing_paths_warned: set[str] = set()

        # Geometry proxy for Sionna runtime bridge.
        self._mesh_aabbs: list[dict[str, Any]] = []
        self._scene_bbox: dict[str, list[float]] | None = None
        self._tx_anchor_source = "unknown"
        self._rx_anchor_source = "unknown"

        # Late-bound Isaac helpers populated in start().
        self._create_new_stage = None
        self._open_stage = None
        self._update_stage = None
        self._create_prim = None
        self._get_world_pose = None
        self._set_world_pose = None
        self._prev_actor_state: dict[str, dict[str, Any]] = {}
        self._camera_adapter: CameraAdapter | None = None

    def _ensure_started(self) -> None:
        if self._simulation_app is None or self._world is None:
            raise RuntimeError("IsaacAdapter not started. Call start() first.")

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._missing_paths_warned:
            return
        self._missing_paths_warned.add(key)
        print(message)

    def _create_fallback_stage(self) -> None:
        """Create a minimal deterministic stage for smoke tests."""
        self._create_prim(
            prim_path="/World",
            prim_type="Xform",
        )

        self._create_prim(
            prim_path=self.tx_prim_path,
            prim_type="Xform",
            position=[0.0, 0.0, 1.5],
            orientation=[1.0, 0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
        )

        self._create_prim(
            prim_path=self.rx_prim_path,
            prim_type="Xform",
            position=[5.0, 0.0, 1.5],
            orientation=[1.0, 0.0, 0.0, 0.0],
            scale=[1.0, 1.0, 1.0],
        )

        if not self.actor_prim_paths:
            self.actor_prim_paths = ["/World/actor_0"]

        for i, actor_path in enumerate(self.actor_prim_paths):
            self._create_prim(
                prim_path=actor_path,
                prim_type="Xform",
                position=[2.5 + float(i), 1.0, 0.0],
                orientation=[1.0, 0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
            )

    def _stage_has_prim(self, prim_path: str) -> bool:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return False
        prim = stage.GetPrimAtPath(prim_path)
        return bool(prim and prim.IsValid())

    def _auto_anchor_from_bbox(self, offset_norm: list[float]) -> list[float]:
        if self._scene_bbox is None:
            return [0.0, 0.0, self.anchor_height_m]

        mn = self._scene_bbox["min_xyz"]
        mx = self._scene_bbox["max_xyz"]
        x = mn[0] + max(0.0, min(1.0, offset_norm[0])) * (mx[0] - mn[0])
        y = mn[1] + max(0.0, min(1.0, offset_norm[1])) * (mx[1] - mn[1])
        z = self.anchor_height_m
        return [float(x), float(y), float(z)]

    def _ensure_tx_rx_anchors(self) -> None:
        tx_exists = self._stage_has_prim(self.tx_prim_path)
        rx_exists = self._stage_has_prim(self.rx_prim_path)

        if tx_exists:
            self._tx_anchor_source = "existing_prim"
        else:
            if self.manual_tx_xyz is not None:
                tx_pos = [float(v) for v in self.manual_tx_xyz]
                self._tx_anchor_source = "manual_xyz"
            else:
                tx_pos = self._auto_anchor_from_bbox(self.tx_offset_norm)
                self._tx_anchor_source = "bbox_auto"
            self._create_prim(
                prim_path=self.tx_prim_path,
                prim_type="Xform",
                position=tx_pos,
                orientation=[1.0, 0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
            )

        if rx_exists:
            self._rx_anchor_source = "existing_prim"
        else:
            if self.manual_rx_xyz is not None:
                rx_pos = [float(v) for v in self.manual_rx_xyz]
                self._rx_anchor_source = "manual_xyz"
            else:
                rx_pos = self._auto_anchor_from_bbox(self.rx_offset_norm)
                self._rx_anchor_source = "bbox_auto"
            self._create_prim(
                prim_path=self.rx_prim_path,
                prim_type="Xform",
                position=rx_pos,
                orientation=[1.0, 0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
            )

    def _ensure_actor_prims(self) -> None:
        if not self.actor_prim_paths:
            return
        for i, actor_path in enumerate(self.actor_prim_paths):
            if self._stage_has_prim(actor_path):
                continue
            self._create_prim(
                prim_path=actor_path,
                prim_type="Xform",
                position=[2.5 + float(i), 1.0, 0.0],
                orientation=[1.0, 0.0, 0.0, 0.0],
                scale=[1.0, 1.0, 1.0],
            )

    def _update_geometry_proxy(self) -> None:
        self._mesh_aabbs = []
        self._scene_bbox = None

        try:
            if not self.scene_usd:
                return
            mesh_limit = max(self.max_meshes, self.geometry_mesh_max_meshes)
            aabbs = extract_mesh_aabbs_from_usd_file(self.scene_usd, max_meshes=mesh_limit)
        except Exception as exc:
            self._warn_once("mesh_extract", f"[IsaacAdapter] WARN: mesh extraction failed: {exc}")
            return

        if not aabbs:
            return

        self._mesh_aabbs = [
            {
                "prim_path": m.prim_path,
                "center_xyz": m.center_xyz,
                "half_extent_xyz": m.half_extent_xyz,
            }
            for m in aabbs
        ]
        self._scene_bbox = compute_global_bbox(aabbs)

    def _read_pose(self, prim_path: str | None) -> dict[str, Any] | None:
        """Read world pose for a prim path in {pos_xyz, quat_wxyz} format."""
        if not prim_path:
            return None

        try:
            pos, quat = self._get_world_pose(prim_path)
        except Exception as exc:  # pragma: no cover - runtime-dependent path errors
            self._warn_once(
                prim_path,
                f"[IsaacAdapter] WARN: pose read failed for '{prim_path}': {exc}",
            )
            return None

        return {
            "prim_path": prim_path,
            "pos_xyz": [float(v) for v in pos],
            "quat_wxyz": [float(v) for v in quat],
        }

    def _apply_pose(self, prim_path: str, pos_xyz: list[float], quat_wxyz: list[float]) -> None:
        if self._set_world_pose is not None:
            self._set_world_pose(prim_path, pos_xyz, quat_wxyz)
            return

        import omni.usd
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return

        xformable = UsdGeom.Xformable(prim)
        translate_op = None
        orient_op = None
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                translate_op = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                orient_op = op
        if translate_op is None:
            translate_op = xformable.AddTranslateOp()
        if orient_op is None:
            orient_op = xformable.AddOrientOp()

        translate_op.Set(Gf.Vec3d(float(pos_xyz[0]), float(pos_xyz[1]), float(pos_xyz[2])))
        orient_op.Set(
            Gf.Quatd(
                float(quat_wxyz[0]),
                Gf.Vec3d(float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])),
            )
        )

    def start(self) -> None:
        """Initialize Isaac Sim app/session and stage."""
        if self._simulation_app is not None:
            print("[IsaacAdapter] start() called while already started")
            return

        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": self.headless})
        isaac_seeded = seed_isaac_runtime(self.seed)

        # Import Isaac APIs only after SimulationApp bootstraps the kernel.
        from isaacsim.core.api import World
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.utils.stage import create_new_stage, open_stage, update_stage
        from isaacsim.core.utils import xforms as isaac_xforms

        self._World = World
        self._create_new_stage = create_new_stage
        self._open_stage = open_stage
        self._update_stage = update_stage
        self._create_prim = create_prim
        self._get_world_pose = isaac_xforms.get_world_pose
        self._set_world_pose = getattr(isaac_xforms, "set_world_pose", None)

        loaded = False
        scene_usd = Path(self.scene_usd) if self.scene_usd else None
        if scene_usd and scene_usd.exists():
            try:
                loaded = bool(self._open_stage(str(scene_usd.resolve())))
            except Exception as exc:  # pragma: no cover - runtime-dependent
                print(f"[IsaacAdapter] WARN: failed to open USD '{scene_usd}': {exc}")
                loaded = False

        if loaded:
            self._stage_source = "usd"
            self._update_stage()
            self._update_geometry_proxy()
            self._ensure_tx_rx_anchors()
            self._ensure_actor_prims()
            self._update_stage()
            self._update_geometry_proxy()
        else:
            self._create_new_stage()
            self._update_stage()
            self._create_fallback_stage()
            self._update_stage()
            self._stage_source = "autogen"
            self._update_geometry_proxy()

        dt = 1.0 / max(self.isaac_fps, 1.0)
        self._world = World(physics_dt=dt, rendering_dt=dt)
        self._world.reset()
        self._world.play()
        self._frame_idx = 0
        self._prev_actor_state = {}
        if self.camera_enabled:
            self._camera_adapter = CameraAdapter(self.config)
            try:
                ok = self._camera_adapter.initialize()
            except Exception as exc:  # pragma: no cover - runtime dependent
                ok = False
                print(f"[IsaacAdapter] WARN: camera initialization crashed: {exc}")
            if not ok:
                err = self._camera_adapter.init_error
                print(f"[IsaacAdapter] WARN: camera disabled due to init failure ({err})")
                self._camera_adapter = None

        print(
            f"[IsaacAdapter] started headless={self.headless} render={self.render} "
            f"stage_source={self._stage_source} meshes={len(self._mesh_aabbs)} "
            f"seed={self.seed} isaac_seeded={isaac_seeded} "
            f"camera_enabled={self.camera_enabled} camera_ready={self._camera_adapter is not None}"
        )

    def step(self) -> None:
        """Advance one simulation tick."""
        self._ensure_started()
        self._world.step(render=self.render)
        self._frame_idx += 1

        if self._actor_motion.enabled and self._actor_motion.update_every_tick:
            timestamp_sim = float(self._world.current_time)
            for pose in self._actor_motion.target_poses(timestamp_sim=timestamp_sim, frame_idx=self._frame_idx):
                try:
                    self._apply_pose(pose.prim_path, pose.position_xyz, pose.orientation_quat_wxyz)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    self._warn_once(
                        f"motion::{pose.prim_path}",
                        f"[IsaacAdapter] WARN: motion apply failed for '{pose.prim_path}': {exc}",
                    )

    def get_sionna_geometry_proxy(self) -> dict[str, Any]:
        """Return stage-derived geometry proxy used by Sionna bridge."""
        return {
            "stage_source": self._stage_source,
            "mesh_aabbs": list(self._mesh_aabbs),
            "scene_bbox": self._scene_bbox,
            "scene_usd": self.scene_usd,
            "geometry_mode": self.geometry_mode,
            "geometry_mesh_format": self.geometry_mesh_format,
            "geometry_mesh_output_dir": self.geometry_mesh_output_dir,
            "geometry_mesh_include_regex": self.geometry_mesh_include_regex,
            "geometry_mesh_exclude_regex": self.geometry_mesh_exclude_regex,
            "geometry_mesh_max_meshes": self.geometry_mesh_max_meshes,
        }

    def capture_rgb(self, frame_idx: int) -> dict[str, Any] | None:
        """Capture one synchronized RGB frame if camera is enabled."""
        self._ensure_started()
        if self._camera_adapter is None:
            return None
        return self._camera_adapter.capture(frame_idx=frame_idx)

    def get_state(self) -> dict:
        """Return current simulation state and tracked poses."""
        self._ensure_started()

        tx_pose = self._read_pose(self.tx_prim_path)
        rx_pose = self._read_pose(self.rx_prim_path)

        actors = []
        actor_poses = []
        timestamp_sim = float(self._world.current_time)
        for prim_path in self.actor_prim_paths:
            pose = self._read_pose(prim_path)
            if pose is not None:
                actors.append(pose)
                pos = [float(v) for v in pose["pos_xyz"]]
                quat = [float(v) for v in pose["quat_wxyz"]]
                prev = self._prev_actor_state.get(prim_path)
                if prev is None:
                    lin_vel = [0.0, 0.0, 0.0]
                    ang_vel = [0.0, 0.0, 0.0]
                else:
                    dt = max(timestamp_sim - float(prev["timestamp_sim"]), 0.0)
                    if dt <= 1e-9:
                        lin_vel = [0.0, 0.0, 0.0]
                        ang_vel = [0.0, 0.0, 0.0]
                    else:
                        prev_pos = prev["pos_xyz"]
                        lin_vel = [(pos[i] - prev_pos[i]) / dt for i in range(3)]
                        ang_vel = angular_velocity_from_quats(prev["quat_wxyz"], quat, dt)

                self._prev_actor_state[prim_path] = {
                    "timestamp_sim": timestamp_sim,
                    "pos_xyz": pos,
                    "quat_wxyz": quat,
                }
                actor_poses.append(
                    {
                        "prim_path": prim_path,
                        "position_xyz": pos,
                        "orientation_quat_wxyz": quat,
                        "velocity_linear_xyz_mps": [float(v) for v in lin_vel],
                        "velocity_angular_xyz_radps": [float(v) for v in ang_vel],
                        "motion_type": self._actor_motion.motion_type_for(prim_path),
                    }
                )

        return {
            "timestamp_sim": timestamp_sim,
            "frame_idx": int(self._frame_idx),
            "tx_pose": tx_pose,
            "rx_pose": rx_pose,
            "actors": actors,
            "actor_poses": actor_poses,
            "stage_source": self._stage_source,
            "is_playing": bool(self._world.is_playing()),
            "tx_anchor_source": self._tx_anchor_source,
            "rx_anchor_source": self._rx_anchor_source,
        }

    def stop(self) -> None:
        """Shutdown Isaac Sim session cleanly."""
        if self._world is not None:
            try:
                if self._world.is_playing():
                    self._world.stop()
            except Exception:
                pass

        if self._World is not None:
            try:
                self._World.clear_instance()
            except Exception:
                pass

        self._world = None
        self._World = None
        self._prev_actor_state = {}
        if self._camera_adapter is not None:
            try:
                self._camera_adapter.close()
            except Exception:
                pass
            self._camera_adapter = None

        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            finally:
                self._simulation_app = None

        print("[IsaacAdapter] stopped")
