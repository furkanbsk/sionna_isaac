"""IRA-backed actor motion integration.

This module provides a thin runtime bridge to
`isaacsim.replicator.agent.core` while keeping imports deferred until
SimulationApp has started.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class IraConfigPaths:
    config_yaml: Path
    command_txt: Path
    robot_command_txt: Path


class IraMotionBackend:
    """Backend for actor motion using IsaacSim Replicator Agent (IRA)."""

    def __init__(self, actor_motion_cfg: dict[str, Any] | None, root_config: dict[str, Any] | None = None):
        actor_motion_cfg = actor_motion_cfg or {}
        self._root_config = root_config or {}
        self.enabled = bool(actor_motion_cfg.get("enabled", False))
        self.update_every_tick = bool(actor_motion_cfg.get("update_every_tick", True))

        ira_cfg = actor_motion_cfg.get("ira", {}) if isinstance(actor_motion_cfg, dict) else {}
        self._ira_cfg = ira_cfg if isinstance(ira_cfg, dict) else {}
        self._backend_enabled = bool(self._ira_cfg.get("enabled", True))

        self._config_path = self._ira_cfg.get("config_path")
        self._auto_generate_config = bool(self._ira_cfg.get("auto_generate_config", True))
        self._auto_generate_commands = bool(self._ira_cfg.get("auto_generate_commands", True))
        self._runtime_dir = Path(str(self._ira_cfg.get("runtime_dir", "isaacsim_sionna/data/runtime/ira")))
        self._scene_asset_path = self._ira_cfg.get("scene_asset_path")

        ui_cfg = self._ira_cfg.get("ui", {}) if isinstance(self._ira_cfg, dict) else {}
        self._ui_cfg = ui_cfg if isinstance(ui_cfg, dict) else {}
        self._enable_ui_in_gui = bool(self._ui_cfg.get("enable_in_gui", False))

        nav_cfg = self._ira_cfg.get("navmesh", {}) if isinstance(self._ira_cfg, dict) else {}
        self._nav_cfg = nav_cfg if isinstance(nav_cfg, dict) else {}
        self._fail_if_missing_navmesh = bool(self._nav_cfg.get("fail_if_missing", True))
        self._auto_bootstrap_navmesh = bool(self._nav_cfg.get("auto_bootstrap", True))
        self._navmesh_bootstrap_parent_prim_path = str(self._nav_cfg.get("bootstrap_parent_prim_path", "/World"))
        self._navmesh_bootstrap_retries = max(1, int(self._nav_cfg.get("bootstrap_retries", 1)))
        self._navmesh_bootstrap_update_steps = max(1, int(self._nav_cfg.get("bootstrap_update_steps", 8)))

        char_cfg = self._ira_cfg.get("character", {}) if isinstance(self._ira_cfg, dict) else {}
        self._char_cfg = char_cfg if isinstance(char_cfg, dict) else {}
        tracked_paths = self._char_cfg.get("tracked_prim_paths", [])
        if isinstance(tracked_paths, list):
            self._tracked_paths = [str(p) for p in tracked_paths if str(p).strip()]
        else:
            self._tracked_paths = []

        self._simulation = None
        self._active = False
        self.init_error: str | None = None

    @property
    def backend_name(self) -> str:
        return "ira"

    @property
    def is_active(self) -> bool:
        return bool(self._active)

    def configured_actor_paths(self) -> list[str]:
        return list(self._tracked_paths)

    def runtime_actor_paths(self) -> list[str]:
        paths = list(self._tracked_paths)
        if not self._active:
            return paths
        for p in self._discover_character_paths():
            if p not in paths:
                paths.append(p)
        return paths

    def motion_type_for(self, prim_path: str) -> str:
        if prim_path in self.runtime_actor_paths():
            return "ira"
        return "none"

    def target_poses(self, timestamp_sim: float, frame_idx: int) -> list[Any]:
        _ = timestamp_sim, frame_idx
        # IRA updates character transforms through its own behavior system.
        return []

    def start(self, *, scene_usd: str | None, headless: bool, seed: int, max_frames: int) -> bool:
        if not self.enabled or not self._backend_enabled:
            return False

        try:
            from isaacsim.core.utils.extensions import enable_extension  # pylint: disable=import-outside-toplevel
            import omni.kit.app  # pylint: disable=import-outside-toplevel

            enable_extension("omni.anim.navigation.core")
            enable_extension("omni.anim.navigation.recast")
            enable_extension("isaacsim.replicator.agent.core")
            if not headless and self._enable_ui_in_gui:
                enable_extension("isaacsim.replicator.agent.ui")
            # Extension startup is async; give nav/recast time to register providers.
            app = omni.kit.app.get_app()
            warmup_steps = max(1, int(self._ira_cfg.get("extension_warmup_steps", 12)))
            for _ in range(warmup_steps):
                app.update()
            # Force-load nav provider before IRA setup triggers navmesh baking.
            import omni.anim.navigation.recast as nav_recast  # pylint: disable=import-outside-toplevel
            recast_iface = None
            try:
                recast_iface = nav_recast.acquire_interface()
            except Exception:
                recast_iface = None
            if recast_iface is None:
                # Give startup a little more time if provider is not ready yet.
                for _ in range(warmup_steps):
                    app.update()
                try:
                    recast_iface = nav_recast.acquire_interface()
                except Exception:
                    recast_iface = None

            from isaacsim.replicator.agent.core.simulation import (  # pylint: disable=import-outside-toplevel
                SimulationManager,
            )

            cfg_path = self._resolve_or_generate_config(scene_usd=scene_usd, seed=seed, max_frames=max_frames)
            sim = SimulationManager()
            # SimulationManager acquires recast interface in __init__; allow it to fully initialize.
            setup_warmup_steps = max(1, int(self._ira_cfg.get("setup_warmup_steps", 16)))
            for _ in range(setup_warmup_steps):
                app.update()
            if not sim.load_config_file(str(cfg_path.resolve())):
                self.init_error = f"ira_config_load_failed:{cfg_path}"
                self._cleanup_runtime_files()
                return False

            navmesh_ready = self._ensure_navmesh_volume()
            if self._fail_if_missing_navmesh and not navmesh_ready:
                self.init_error = "ira_navmesh_bootstrap_failed"
                self._cleanup_runtime_files()
                return False

            sim.set_up_simulation_from_config_file()
            ok = self._wait_for_setup(sim, timeout_s=float(self._ira_cfg.get("setup_timeout_s", 15.0)))
            if not ok:
                self.init_error = "ira_setup_timeout"
                self._cleanup_runtime_files()
                return False

            navmesh_ready = self._stage_has_navmesh_volume()
            if self._fail_if_missing_navmesh and not navmesh_ready:
                self.init_error = "ira_navmesh_missing"
                self._cleanup_runtime_files()
                return False

            self._simulation = sim
            self._active = True
            self._tracked_paths = self.runtime_actor_paths()
            return True
        except Exception as exc:  # pragma: no cover - runtime dependent
            self.init_error = f"ira_start_exception:{exc}"
            self._active = False
            self._cleanup_runtime_files()
            return False

    def _ensure_navmesh_volume(self) -> bool:
        try:
            import omni.kit.app  # pylint: disable=import-outside-toplevel
            import omni.kit.commands  # pylint: disable=import-outside-toplevel
            import omni.usd  # pylint: disable=import-outside-toplevel
            from pxr import Gf, Sdf, UsdGeom  # pylint: disable=import-outside-toplevel
            import NavSchema  # pylint: disable=import-outside-toplevel

            if self._stage_has_navmesh_volume():
                return True
            if not self._auto_bootstrap_navmesh:
                return False

            app = omni.kit.app.get_app()
            for _ in range(self._navmesh_bootstrap_retries):
                try:
                    omni.kit.commands.execute(
                        "CreateNavMeshVolumeCommand",
                        parent_prim_path=self._navmesh_bootstrap_parent_prim_path,
                    )
                except Exception:
                    # Fall back to direct USD API creation when command registration is unavailable.
                    usd_ctx = omni.usd.get_context()
                    stage = usd_ctx.get_stage()
                    if stage is not None:
                        parent = self._navmesh_bootstrap_parent_prim_path.rstrip("/") or "/World"
                        base = f"{parent}/NavMeshVolume"
                        prim_path_str = omni.usd.get_stage_next_free_path(stage, base, True)
                        prim_path = Sdf.Path(prim_path_str)
                        NavSchema.NavMeshVolume.Define(stage, prim_path)
                        prim = stage.GetPrimAtPath(prim_path)

                        half_extent = 0.5 / max(UsdGeom.GetStageMetersPerUnit(stage), 1e-9)
                        boundable = UsdGeom.Boundable(prim)
                        extent_attr = boundable.GetExtentAttr()
                        if extent_attr:
                            extent_attr.Set(
                                [(-half_extent, -half_extent, -half_extent), (half_extent, half_extent, half_extent)]
                            )

                        world_bound = usd_ctx.compute_path_world_bounding_box(Sdf.Path.absoluteRootPath.pathString)
                        world_range = Gf.Range3d(Gf.Vec3d(*world_bound[0]), Gf.Vec3d(*world_bound[1]))
                        mid_point = world_range.GetMidpoint()
                        dimension = world_range.GetSize()
                        min_size = 400.0
                        for i in range(3):
                            if dimension[i] < min_size:
                                dimension[i] = min_size
                            dimension[i] += 50.0
                        translate = Gf.Matrix4d(1.0)
                        translate.SetTranslate(mid_point)
                        scale = Gf.Matrix4d(1.0)
                        scale.SetScale(dimension)
                        xform = scale * translate
                        parent_world_xform = Gf.Matrix4d(
                            *usd_ctx.compute_path_world_transform(prim_path.GetParentPath().pathString)
                        )
                        xform *= parent_world_xform.GetInverse()
                        xformable = UsdGeom.Xformable(prim)
                        xform_op = xformable.MakeMatrixXform()
                        xform_op.Set(xform)
                for _ in range(self._navmesh_bootstrap_update_steps):
                    app.update()
                if self._stage_has_navmesh_volume():
                    return True
            return self._stage_has_navmesh_volume()
        except Exception:
            return False

    def _stage_has_navmesh_volume(self) -> bool:
        try:
            import omni.usd  # pylint: disable=import-outside-toplevel

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return False
            preferred_parent = self._navmesh_bootstrap_parent_prim_path.rstrip("/")
            if not preferred_parent:
                preferred_parent = "/World"
            preferred_prefix = preferred_parent + "/"
            any_volume = False
            for prim in stage.Traverse():
                if not prim.IsValid():
                    continue
                if prim.GetTypeName() == "NavMeshVolume":
                    any_volume = True
                    p = str(prim.GetPath())
                    if p == preferred_parent or p.startswith(preferred_prefix):
                        return True
            # A volume outside the expected parent is treated as missing so
            # bootstrap can create one where IRA expects it.
            _ = any_volume
            return False
        except Exception:
            return False

    def stop(self) -> None:
        self._simulation = None
        self._active = False

    def _cleanup_runtime_files(self) -> None:
        """Remove auto-generated config/command files on startup failure."""
        for name in ("ira_runtime.yaml", "command.txt", "robot_command.txt"):
            path = self._runtime_dir / name
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                logger.warning("Failed to clean up runtime file %s: %s", path, exc)

    @staticmethod
    def _dispose_subscription(sub: Any) -> None:
        """Best-effort unsubscribe from IRA callback."""
        if sub is None:
            return
        # Omni subscriptions expose .unsubscribe() or are callable disposers.
        try:
            if hasattr(sub, "unsubscribe") and callable(sub.unsubscribe):
                sub.unsubscribe()
                return
        except Exception:
            pass
        try:
            if callable(sub):
                sub()
        except Exception:
            pass

    def _wait_for_setup(self, sim: Any, timeout_s: float) -> bool:
        try:
            import omni.kit.app  # pylint: disable=import-outside-toplevel

            done = {"ok": False}

            def _on_done(_event):
                done["ok"] = True

            sub = sim.register_set_up_simulation_done_callback(_on_done)
            app = omni.kit.app.get_app()
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < timeout_s:
                app.update()
                if done["ok"]:
                    self._dispose_subscription(sub)
                    return True
            self._dispose_subscription(sub)
            return False
        except Exception:
            return False

    def _resolve_or_generate_config(self, *, scene_usd: str | None, seed: int, max_frames: int) -> Path:
        if self._config_path:
            cfg_path = Path(str(self._config_path))
            if cfg_path.exists():
                return cfg_path

        if not self._auto_generate_config:
            raise RuntimeError("ira config_path missing and auto_generate_config=false")

        self._runtime_dir.mkdir(parents=True, exist_ok=True)
        cfg_paths = self._build_runtime_files(scene_usd=scene_usd, seed=seed, max_frames=max_frames)
        return cfg_paths.config_yaml

    def _build_runtime_files(self, *, scene_usd: str | None, seed: int, max_frames: int) -> IraConfigPaths:
        character_count = int(self._char_cfg.get("num", 1))
        filters_raw = self._char_cfg.get("filters", "")
        filters = "" if filters_raw is None else str(filters_raw)
        asset_path = self._char_cfg.get("asset_path", "")
        if not asset_path:
            # Keep explicit fallback asset path to avoid empty-path character list failures.
            asset_path = (
                "http://omniverse-content-production.s3-us-west-2.amazonaws.com/"
                "Assets/Isaac/4.5/Isaac/People/Characters/"
            )

        sim_frames = int(self._ira_cfg.get("simulation_length_frames", max_frames))
        scene_path = str(self._scene_asset_path or scene_usd or "")
        if not scene_path:
            raise RuntimeError("IRA needs scene usd path (isaac.actor_motion.ira.scene_asset_path or scenario.scene_usd)")

        cmd_path = self._runtime_dir / "command.txt"
        robot_cmd_path = self._runtime_dir / "robot_command.txt"
        cfg_yaml = self._runtime_dir / "ira_runtime.yaml"

        if self._auto_generate_commands:
            commands = self._generate_default_commands(character_count)
            cmd_path.write_text("\n".join(commands) + "\n", encoding="utf-8")
            robot_cmd_path.write_text("\n", encoding="utf-8")

        payload = {
            "isaacsim.replicator.agent": {
                "version": "0.5.1",
                "global": {
                    "seed": int(seed),
                    "simulation_length": int(sim_frames),
                },
                "scene": {
                    "asset_path": scene_path,
                },
                "sensor": {
                    "camera_num": 0,
                },
                "character": {
                    "asset_path": str(asset_path),
                    "command_file": str(cmd_path.resolve()),
                    "filters": filters,
                    "num": int(character_count),
                },
                "robot": {
                    "command_file": str(robot_cmd_path.resolve()),
                    "nova_carter_num": 0,
                    "transporter_num": 0,
                    "write_data": False,
                },
                "replicator": {
                    "writer": "IRABasicWriter",
                    "parameters": {
                        "rgb": False,
                        "camera_params": False,
                    },
                },
            }
        }
        cfg_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

        return IraConfigPaths(config_yaml=cfg_yaml, command_txt=cmd_path, robot_command_txt=robot_cmd_path)

    def _generate_default_commands(self, character_count: int) -> list[str]:
        # Command format observed in ISAAC IRA tests: "Character GoTo x y z yaw"
        cmds: list[str] = []
        isaac_cfg = self._root_config.get("isaac", {}) if isinstance(self._root_config, dict) else {}
        cam_cfg = isaac_cfg.get("camera", {}) if isinstance(isaac_cfg, dict) else {}
        target = cam_cfg.get("target_xyz", [0.0, 0.0, 1.5])
        if not isinstance(target, (list, tuple)) or len(target) != 3:
            target = [0.0, 0.0, 1.5]
        tx, ty, tz = float(target[0]), float(target[1]), float(target[2])
        walk_z = max(0.0, tz - 1.5)
        names = ["Character"] + [f"Character_{i:02d}" for i in range(1, max(character_count, 1))]
        for i in range(max(character_count, 1)):
            name = names[i] if i < len(names) else f"Character_{i:02d}"
            # Deterministic short loop command sequence near camera target.
            cmds.append(f"{name} Idle 1")
            cmds.append(f"{name} GoTo {tx:.3f} {ty:.3f} {walk_z:.3f} 0")
            cmds.append(f"{name} Idle 1")
            cmds.append(f"{name} GoTo {tx + 2.0:.3f} {ty:.3f} {walk_z:.3f} 0")
            cmds.append(f"{name} Idle 1")
            cmds.append(f"{name} GoTo {tx - 2.0:.3f} {ty:.3f} {walk_z:.3f} 0")
        return cmds

    def _discover_character_paths(self) -> list[str]:
        try:
            from isaacsim.replicator.agent.core.stage_util import (  # pylint: disable=import-outside-toplevel
                CharacterUtil,
            )

            out = []
            for prim in CharacterUtil.get_characters_root_in_stage(count=-1, count_invisible=True):
                p = str(prim.GetPath())
                if p:
                    out.append(p)
            return out
        except Exception as exc:
            logger.warning("Character path discovery failed: %s", exc)
            return []
