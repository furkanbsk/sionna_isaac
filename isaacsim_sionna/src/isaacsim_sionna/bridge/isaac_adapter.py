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

from isaacsim_sionna.bridge.usd_to_sionna import compute_global_bbox, extract_mesh_aabbs_from_usd_file


class IsaacAdapter:
    """Adapter for stepping Isaac Sim and exporting pose state."""

    def __init__(self, config: dict):
        self.config = config

        runtime_cfg = config.get("runtime", {})
        isaac_cfg = config.get("isaac", {})
        scenario_cfg = config.get("scenario", {})
        prim_cfg = isaac_cfg.get("prim_paths", {})
        anchor_cfg = isaac_cfg.get("anchors", {})

        self.isaac_fps = float(runtime_cfg.get("isaac_fps", 60.0))
        self.headless = bool(isaac_cfg.get("headless", True))
        self.render = bool(isaac_cfg.get("render", False))
        self.scene_usd = scenario_cfg.get("scene_usd")

        self.tx_prim_path = prim_cfg.get("tx", "/World/tx")
        self.rx_prim_path = prim_cfg.get("rx", "/World/rx")
        self.actor_prim_paths = list(prim_cfg.get("actors", ["/World/actor_0"]))

        self.auto_anchor_from_bbox = bool(anchor_cfg.get("auto_from_bbox", True))
        self.anchor_height_m = float(anchor_cfg.get("height_m", 1.5))
        self.tx_offset_norm = [float(v) for v in anchor_cfg.get("tx_offset_norm", [0.2, 0.2])]
        self.rx_offset_norm = [float(v) for v in anchor_cfg.get("rx_offset_norm", [0.8, 0.8])]
        self.manual_tx_xyz = anchor_cfg.get("manual_tx_xyz")
        self.manual_rx_xyz = anchor_cfg.get("manual_rx_xyz")
        self.max_meshes = int(anchor_cfg.get("max_meshes", 256))

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

    def _update_geometry_proxy(self) -> None:
        self._mesh_aabbs = []
        self._scene_bbox = None

        try:
            if not self.scene_usd:
                return
            aabbs = extract_mesh_aabbs_from_usd_file(self.scene_usd, max_meshes=self.max_meshes)
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

    def start(self) -> None:
        """Initialize Isaac Sim app/session and stage."""
        if self._simulation_app is not None:
            print("[IsaacAdapter] start() called while already started")
            return

        from isaacsim import SimulationApp

        self._simulation_app = SimulationApp({"headless": self.headless})

        # Import Isaac APIs only after SimulationApp bootstraps the kernel.
        from isaacsim.core.api import World
        from isaacsim.core.utils.prims import create_prim
        from isaacsim.core.utils.stage import create_new_stage, open_stage, update_stage
        from isaacsim.core.utils.xforms import get_world_pose

        self._World = World
        self._create_new_stage = create_new_stage
        self._open_stage = open_stage
        self._update_stage = update_stage
        self._create_prim = create_prim
        self._get_world_pose = get_world_pose

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

        print(
            f"[IsaacAdapter] started headless={self.headless} render={self.render} "
            f"stage_source={self._stage_source} meshes={len(self._mesh_aabbs)}"
        )

    def step(self) -> None:
        """Advance one simulation tick."""
        self._ensure_started()
        self._world.step(render=self.render)
        self._frame_idx += 1

    def get_sionna_geometry_proxy(self) -> dict[str, Any]:
        """Return stage-derived geometry proxy used by Sionna bridge."""
        return {
            "stage_source": self._stage_source,
            "mesh_aabbs": list(self._mesh_aabbs),
            "scene_bbox": self._scene_bbox,
            "scene_usd": self.scene_usd,
        }

    def get_state(self) -> dict:
        """Return current simulation state and tracked poses."""
        self._ensure_started()

        tx_pose = self._read_pose(self.tx_prim_path)
        rx_pose = self._read_pose(self.rx_prim_path)

        actors = []
        for prim_path in self.actor_prim_paths:
            pose = self._read_pose(prim_path)
            if pose is not None:
                actors.append(pose)

        return {
            "timestamp_sim": float(self._world.current_time),
            "frame_idx": int(self._frame_idx),
            "tx_pose": tx_pose,
            "rx_pose": rx_pose,
            "actors": actors,
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

        if self._simulation_app is not None:
            try:
                self._simulation_app.close()
            finally:
                self._simulation_app = None

        print("[IsaacAdapter] stopped")
