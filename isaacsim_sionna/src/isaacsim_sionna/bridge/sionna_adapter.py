"""Sionna RT adapter with configurable geometry translation modes."""

from __future__ import annotations

from pathlib import Path
import math
import time

import numpy as np

from isaacsim_sionna.bridge.usd_mesh_export import extract_mesh_primitives, export_meshes_to_ply
from isaacsim_sionna.bridge.usd_to_sionna import MeshAabb, build_sionna_xml_from_aabbs, build_sionna_xml_from_mesh_files


class SionnaAdapter:
    """Channel engine using Sionna RT PathSolver with runtime scene XML."""

    def __init__(self, config: dict):
        self.config = config
        self.radio_cfg = config.get("radio", {})
        self.runtime_cfg = config.get("runtime", {})
        self.scenario_cfg = config.get("scenario", {})
        self.isaac_cfg = config.get("isaac", {})
        self.project_cfg = config.get("project", {})

        self.carrier_hz = float(self.radio_cfg.get("carrier_hz", 3.5e9))
        self.bandwidth_hz = float(self.radio_cfg.get("bandwidth_hz", 20e6))
        self.num_subcarriers = int(self.radio_cfg.get("num_subcarriers", 256))
        self.max_depth = int(self.radio_cfg.get("max_depth", 5))
        self.samples_per_src = int(float(self.radio_cfg.get("samples_per_src", 100000)))
        self.base_seed = int(self.project_cfg.get("seed", 42))
        self.solver_seed_strategy = str(self.project_cfg.get("solver_seed_strategy", "frame_offset")).lower()
        self.csi_method = str(self.radio_cfg.get("csi_method", "explicit_cir_to_ofdm")).strip().lower()
        prop_cfg = self.radio_cfg.get("propagation", {}) if isinstance(self.radio_cfg, dict) else {}
        self.propagation_cfg = prop_cfg if isinstance(prop_cfg, dict) else {}
        self.propagation_flags = {
            "los": bool(self.propagation_cfg.get("los", True)),
            "specular_reflection": bool(self.propagation_cfg.get("specular_reflection", True)),
            "diffuse_reflection": bool(self.propagation_cfg.get("diffuse_reflection", True)),
            "refraction": bool(self.propagation_cfg.get("refraction", True)),
            "diffraction": bool(self.propagation_cfg.get("diffraction", True)),
            "edge_diffraction": bool(self.propagation_cfg.get("edge_diffraction", True)),
            "diffraction_lit_region": bool(self.propagation_cfg.get("diffraction_lit_region", True)),
        }

        self._sionna_loaded = False
        self._PathSolver = None
        self._PlanarArray = None
        self._Receiver = None
        self._Transmitter = None
        self._load_scene = None
        self._subcarrier_frequencies = None

        self._state: dict | None = None
        self._scene = None
        self._path_solver = None
        self._frequencies = None
        self._mesh_count = 0
        self._mesh_file_count = 0
        self._geometry_mode = "aabb"
        self._dynamic_actor_paths: list[str] = []
        self._dynamic_actor_proxy_map: dict[str, str] = {}
        self._actor_proxy_half_extent = [0.25, 0.25, 0.9]
        self._init_metrics: dict[str, float] = {"geometry_prep_ms": 0.0}
        self._warned_keys: set[str] = set()

    def _ensure_sionna_runtime_loaded(self) -> None:
        """Import Sionna RT lazily after SimulationApp startup."""
        if self._sionna_loaded:
            return
        from sionna.rt import (  # pylint: disable=import-outside-toplevel
            PathSolver,
            PlanarArray,
            Receiver,
            Transmitter,
            load_scene,
            subcarrier_frequencies,
        )

        self._PathSolver = PathSolver
        self._PlanarArray = PlanarArray
        self._Receiver = Receiver
        self._Transmitter = Transmitter
        self._load_scene = load_scene
        self._subcarrier_frequencies = subcarrier_frequencies
        self._path_solver = PathSolver()
        self._sionna_loaded = True

    def _setup_radio_scene(self) -> None:
        if not self._sionna_loaded:
            raise RuntimeError("Sionna runtime not loaded")

        self._scene.frequency = float(self.carrier_hz)

        array = self._PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        self._scene.tx_array = array
        self._scene.rx_array = array

        tx = self._Transmitter(name="tx", position=[0.0, 0.0, 1.5])
        rx = self._Receiver(name="rx", position=[1.0, 0.0, 1.5])
        self._scene.add(tx)
        self._scene.add(rx)

        subcarrier_spacing = self.bandwidth_hz / self.num_subcarriers
        self._frequencies = self._subcarrier_frequencies(self.num_subcarriers, subcarrier_spacing)

    @staticmethod
    def _quat_wxyz_to_euler_xyz(quat_wxyz: list[float]) -> list[float]:
        if not isinstance(quat_wxyz, (list, tuple)) or len(quat_wxyz) != 4:
            return [0.0, 0.0, 0.0]
        w, x, y, z = [float(v) for v in quat_wxyz]

        # Roll (x-axis rotation)
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)

        # Pitch (y-axis rotation)
        t2 = 2.0 * (w * y - z * x)
        t2 = max(-1.0, min(1.0, t2))
        pitch = math.asin(t2)

        # Yaw (z-axis rotation)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return [roll, pitch, yaw]

    def _resolve_dynamic_actor_paths(self) -> list[str]:
        actor_motion_cfg = (self.isaac_cfg.get("actor_motion") or {}) if isinstance(self.isaac_cfg, dict) else {}
        enabled = bool(actor_motion_cfg.get("enabled", False))
        if not enabled:
            return []

        actors_cfg = actor_motion_cfg.get("actors", [])
        if not isinstance(actors_cfg, list):
            return []

        paths: list[str] = []
        for item in actors_cfg:
            if not isinstance(item, dict):
                continue
            prim_path = str(item.get("prim_path", "")).strip()
            if prim_path and prim_path not in paths:
                paths.append(prim_path)
        return paths

    def _configure_actor_proxies(self) -> None:
        self._dynamic_actor_proxy_map = {}
        if self._scene is None:
            return

        for i, prim_path in enumerate(self._dynamic_actor_paths):
            proxy_id = f"actor_proxy_{i}"
            self._dynamic_actor_proxy_map[prim_path] = proxy_id
            obj = self._scene.get(proxy_id)
            if obj is None:
                continue
            obj.position = [0.0, 0.0, -100.0]
            if hasattr(obj, "scaling"):
                obj.scaling = list(self._actor_proxy_half_extent)

    def initialize(self, geometry_proxy: dict | None = None) -> None:
        """Build runtime scene XML from geometry proxy and initialize solvers."""
        t_init = time.perf_counter()
        self._ensure_sionna_runtime_loaded()

        if self.num_subcarriers < 1:
            raise ValueError("radio.num_subcarriers must be >= 1")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("radio.bandwidth_hz must be > 0")
        if self.carrier_hz <= 0.0:
            raise ValueError("radio.carrier_hz must be > 0")

        if geometry_proxy is None:
            raise ValueError("geometry_proxy is required for SionnaAdapter.initialize")

        geometry_cfg = self.isaac_cfg.get("geometry", {})
        mesh_cfg = geometry_cfg.get("mesh", {})

        mode_proxy = str(geometry_proxy.get("geometry_mode", "")).strip().lower()
        mode_cfg = str(geometry_cfg.get("mode", "aabb")).strip().lower()
        self._geometry_mode = mode_proxy if mode_proxy else mode_cfg

        cache_xml = self.scenario_cfg.get(
            "scene_sionna_cache",
            "isaacsim_sionna/data/scenes_sionna/cache/runtime_scene.xml",
        )
        cache_xml_path = Path(cache_xml)
        self._dynamic_actor_paths = self._resolve_dynamic_actor_paths()
        actor_motion_cfg = self.isaac_cfg.get("actor_motion", {})
        half_extent_cfg = actor_motion_cfg.get("proxy_half_extent_xyz", [0.25, 0.25, 0.9])
        if isinstance(half_extent_cfg, (list, tuple)) and len(half_extent_cfg) == 3:
            self._actor_proxy_half_extent = [float(half_extent_cfg[0]), float(half_extent_cfg[1]), float(half_extent_cfg[2])]

        if self._geometry_mode == "mesh":
            scene_usd = geometry_proxy.get("scene_usd")
            if not scene_usd:
                raise RuntimeError("geometry_mode=mesh requires scene_usd in geometry_proxy")

            mesh_format = str(geometry_proxy.get("geometry_mesh_format") or mesh_cfg.get("format", "ply")).lower()
            if mesh_format != "ply":
                raise ValueError(f"Unsupported mesh format: {mesh_format}")

            output_dir = geometry_proxy.get("geometry_mesh_output_dir") or mesh_cfg.get(
                "output_dir", "isaacsim_sionna/data/scenes_sionna/runtime_meshes"
            )
            include_regex = geometry_proxy.get("geometry_mesh_include_regex")
            if include_regex is None:
                include_regex = mesh_cfg.get("include_regex")
            exclude_regex = geometry_proxy.get("geometry_mesh_exclude_regex")
            if exclude_regex is None:
                exclude_regex = mesh_cfg.get("exclude_regex")

            max_meshes = int(
                geometry_proxy.get("geometry_mesh_max_meshes") or mesh_cfg.get("max_meshes", 256)
            )
            usd_meshes = extract_mesh_primitives(
                scene_usd=scene_usd,
                max_meshes=max_meshes,
                include_regex=include_regex,
                exclude_regex=exclude_regex,
            )
            if not usd_meshes:
                raise RuntimeError("No USD mesh primitives extracted for geometry_mode=mesh")

            mesh_refs = export_meshes_to_ply(usd_meshes, output_dir=output_dir)
            build_sionna_xml_from_mesh_files(
                mesh_refs,
                cache_xml_path,
                dynamic_actor_count=len(self._dynamic_actor_paths),
            )
            self._scene = self._load_scene(str(cache_xml_path.resolve()), merge_shapes=False)

            self._mesh_count = len(usd_meshes)
            self._mesh_file_count = len(mesh_refs)

        elif self._geometry_mode == "aabb":
            mesh_dicts = geometry_proxy.get("mesh_aabbs", [])
            mesh_aabbs = [
                MeshAabb(
                    prim_path=str(m["prim_path"]),
                    center_xyz=[float(v) for v in m["center_xyz"]],
                    half_extent_xyz=[float(v) for v in m["half_extent_xyz"]],
                )
                for m in mesh_dicts
            ]
            if not mesh_aabbs:
                raise RuntimeError(
                    "No mesh AABBs extracted from USD stage. "
                    "Cannot build real-geometry Sionna scene."
                )

            build_sionna_xml_from_aabbs(
                mesh_aabbs,
                cache_xml_path,
                dynamic_actor_count=len(self._dynamic_actor_paths),
            )
            self._scene = self._load_scene(str(cache_xml_path.resolve()), merge_shapes=False)

            for i, mesh in enumerate(mesh_aabbs):
                obj = self._scene.get(f"mesh_box_{i}")
                obj.position = mesh.center_xyz
                obj.scaling = mesh.half_extent_xyz

            self._mesh_count = len(mesh_aabbs)
            self._mesh_file_count = 0

        else:
            raise ValueError(f"Unsupported geometry mode: {self._geometry_mode}")

        self._setup_radio_scene()
        self._configure_actor_proxies()
        self._init_metrics["geometry_prep_ms"] = (time.perf_counter() - t_init) * 1000.0

        print(
            "[SionnaAdapter] initialized "
            f"mode={self._geometry_mode} "
            f"carrier_hz={self.carrier_hz:.3e} "
            f"bandwidth_hz={self.bandwidth_hz:.3e} "
            f"num_subcarriers={self.num_subcarriers} "
            f"mesh_count={self._mesh_count} mesh_files={self._mesh_file_count}"
        )

    def get_init_metrics(self) -> dict[str, float]:
        return dict(self._init_metrics)

    def update_dynamic_state(self, state: dict) -> None:
        """Update TX/RX transforms in the Sionna scene."""
        self._state = state
        if self._scene is None:
            return

        tx_pose = state.get("tx_pose") or {}
        rx_pose = state.get("rx_pose") or {}
        tx_pos = tx_pose.get("pos_xyz")
        rx_pos = rx_pose.get("pos_xyz")

        if tx_pos is not None:
            self._scene.get("tx").position = tx_pos
        if rx_pos is not None:
            self._scene.get("rx").position = rx_pos

        for actor in (state.get("actor_poses") or []):
            prim_path = actor.get("prim_path")
            if not prim_path:
                continue
            proxy_id = self._dynamic_actor_proxy_map.get(str(prim_path))
            if not proxy_id:
                continue
            obj = self._scene.get(proxy_id)
            if obj is None:
                continue
            pos = actor.get("position_xyz")
            if pos is not None:
                obj.position = pos
            quat = actor.get("orientation_quat_wxyz")
            if quat is not None and hasattr(obj, "orientation"):
                obj.orientation = self._quat_wxyz_to_euler_xyz(quat)

    def _resolve_solver_seed(self, frame_idx: int | None) -> int:
        if self.solver_seed_strategy == "fixed":
            return int(self.base_seed)
        if frame_idx is None:
            return int(self.base_seed)
        return int(self.base_seed + int(frame_idx))

    def _warn_once(self, key: str, msg: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        print(msg)

    def _compute_paths_with_fallback(self, solver_kwargs: dict) -> object:
        """Call PathSolver while dropping unsupported kwargs for API compatibility."""
        kwargs = dict(solver_kwargs)
        while True:
            try:
                return self._path_solver(self._scene, **kwargs)
            except TypeError as exc:
                msg = str(exc)
                marker = "got an unexpected keyword argument '"
                if marker not in msg:
                    raise
                bad_key = msg.split(marker, 1)[1].split("'", 1)[0]
                if bad_key not in kwargs:
                    raise
                kwargs.pop(bad_key, None)
                self._warn_once(
                    f"unsupported::{bad_key}",
                    f"[SionnaAdapter] WARN: PathSolver does not support '{bad_key}'. Falling back without it.",
                )

    def _cir_to_ofdm_explicit(self, a_flat: np.ndarray, tau_flat: np.ndarray) -> np.ndarray:
        """Compute OFDM CSI from CIR using explicit per-subcarrier phase terms."""
        if a_flat.size == 0 or tau_flat.size == 0:
            return np.zeros((self.num_subcarriers,), dtype=np.complex128)
        if a_flat.size != tau_flat.size:
            n = min(int(a_flat.size), int(tau_flat.size))
            a_flat = a_flat[:n]
            tau_flat = tau_flat[:n]

        subcarrier_spacing = float(self.bandwidth_hz) / float(self.num_subcarriers)
        k = np.arange(self.num_subcarriers, dtype=np.float64) - (float(self.num_subcarriers) / 2.0)
        f_sub = k * subcarrier_spacing
        phi = -2.0 * np.pi * np.outer(tau_flat.astype(np.float64), f_sub)
        return np.sum(a_flat[:, None] * np.exp(1j * phi), axis=0)

    def compute_snapshot(self, frame_idx: int | None = None) -> dict:
        """Return one JSON-serializable CIR and CSI snapshot from PathSolver."""
        if self._scene is None or self._frequencies is None or self._path_solver is None:
            raise RuntimeError("SionnaAdapter not initialized. Call initialize() first.")

        tx_pose = (self._state or {}).get("tx_pose")
        rx_pose = (self._state or {}).get("rx_pose")
        if tx_pose is None or rx_pose is None:
            solver_seed = self._resolve_solver_seed(frame_idx)
            return {
                "status": "missing_pose",
                "a_re": [],
                "a_im": [],
                "tau_s": [],
                "csi_re": [],
                "csi_im": [],
                "distance_m": None,
                "num_paths": 0,
                "path_types_present": [],
                "mesh_boxes": self._mesh_count,
                "mesh_file_count": self._mesh_file_count,
                "geometry_mode": self._geometry_mode,
                "solver_seed": solver_seed,
                "propagation": dict(self.propagation_flags),
                "csi_method": self.csi_method,
            }
        solver_seed = self._resolve_solver_seed(frame_idx)

        solver_kwargs = {
            "max_depth": self.max_depth,
            "samples_per_src": self.samples_per_src,
            "los": self.propagation_flags["los"],
            "specular_reflection": self.propagation_flags["specular_reflection"],
            "diffuse_reflection": self.propagation_flags["diffuse_reflection"],
            "refraction": self.propagation_flags["refraction"],
            "diffraction": self.propagation_flags["diffraction"],
            "edge_diffraction": self.propagation_flags["edge_diffraction"],
            "diffraction_lit_region": self.propagation_flags["diffraction_lit_region"],
            "seed": solver_seed,
        }
        paths = self._compute_paths_with_fallback(solver_kwargs)

        a, tau = paths.cir(out_type="numpy")
        a_flat = np.asarray(a).reshape(-1)
        tau_flat = np.asarray(tau).reshape(-1)

        if self.csi_method == "sionna_cfr":
            h = paths.cfr(self._frequencies, out_type="numpy")
            h_arr = np.asarray(h)
            if h_arr.ndim >= 1:
                csi = h_arr.reshape(-1, h_arr.shape[-1])[0]
            else:
                csi = h_arr
        else:
            if self.csi_method != "explicit_cir_to_ofdm":
                self._warn_once(
                    f"csi_method::{self.csi_method}",
                    f"[SionnaAdapter] WARN: Unknown radio.csi_method='{self.csi_method}'. Falling back to explicit_cir_to_ofdm.",
                )
            csi = self._cir_to_ofdm_explicit(np.asarray(a_flat), np.asarray(tau_flat))

        tx = np.asarray(tx_pose["pos_xyz"], dtype=np.float64)
        rx = np.asarray(rx_pose["pos_xyz"], dtype=np.float64)
        distance_m = float(np.linalg.norm(rx - tx))
        path_types_present = [name for name, enabled in self.propagation_flags.items() if enabled]

        return {
            "status": "ok",
            "a_re": np.real(a_flat).astype(np.float64).tolist(),
            "a_im": np.imag(a_flat).astype(np.float64).tolist(),
            "tau_s": np.real(tau_flat).astype(np.float64).tolist(),
            "csi_re": np.real(csi).astype(np.float64).tolist(),
            "csi_im": np.imag(csi).astype(np.float64).tolist(),
            "distance_m": distance_m,
            "num_paths": int(a_flat.size),
            "path_types_present": path_types_present,
            "mesh_boxes": self._mesh_count,
            "mesh_file_count": self._mesh_file_count,
            "geometry_mode": self._geometry_mode,
            "solver_seed": solver_seed,
            "propagation": dict(self.propagation_flags),
            "csi_method": self.csi_method,
        }
