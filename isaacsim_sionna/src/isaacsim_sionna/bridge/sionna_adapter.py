"""Sionna RT adapter with configurable geometry translation modes."""

from __future__ import annotations

import logging
from pathlib import Path
import math
import time

import numpy as np

logger = logging.getLogger(__name__)

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
        self._visualize_rays = bool(self.isaac_cfg.get("visualize_rays", False))
        ray_viz_cfg = self.isaac_cfg.get("ray_viz", {}) if isinstance(self.isaac_cfg, dict) else {}
        self._max_paths_to_draw = int(ray_viz_cfg.get("max_paths_to_draw", 128))
        self._log_path_stats = bool(ray_viz_cfg.get("log_path_stats", True))
        self._last_path_geometry: list[dict] = []
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

        antenna_cfg = self.radio_cfg.get("antenna", {}) if isinstance(self.radio_cfg, dict) else {}
        array = self._PlanarArray(
            num_rows=int(antenna_cfg.get("num_rows", 1)),
            num_cols=int(antenna_cfg.get("num_cols", 1)),
            vertical_spacing=float(antenna_cfg.get("vertical_spacing", 0.5)),
            horizontal_spacing=float(antenna_cfg.get("horizontal_spacing", 0.5)),
            pattern=str(antenna_cfg.get("pattern", "iso")),
            polarization=str(antenna_cfg.get("polarization", "V")),
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
                if obj is None:
                    logger.warning("mesh_box_%d not found in Sionna scene; skipping", i)
                    continue
                obj.position = mesh.center_xyz
                obj.scaling = mesh.half_extent_xyz

            self._mesh_count = len(mesh_aabbs)
            self._mesh_file_count = 0

        else:
            raise ValueError(f"Unsupported geometry mode: {self._geometry_mode}")

        self._setup_radio_scene()
        self._configure_actor_proxies()
        self._init_metrics["geometry_prep_ms"] = (time.perf_counter() - t_init) * 1000.0

        logger.info(
            "initialized mode=%s carrier_hz=%.3e bandwidth_hz=%.3e "
            "num_subcarriers=%d mesh_count=%d mesh_files=%d",
            self._geometry_mode, self.carrier_hz, self.bandwidth_hz,
            self.num_subcarriers, self._mesh_count, self._mesh_file_count,
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
            tx_obj = self._scene.get("tx")
            if tx_obj is not None:
                tx_obj.position = tx_pos
            else:
                logger.warning("TX object not found in Sionna scene")
        if rx_pos is not None:
            rx_obj = self._scene.get("rx")
            if rx_obj is not None:
                rx_obj.position = rx_pos
            else:
                logger.warning("RX object not found in Sionna scene")

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
        logger.warning(msg)

    def _compute_paths_with_fallback(self, solver_kwargs: dict) -> object:
        """Call PathSolver while dropping unsupported kwargs for API compatibility."""
        kwargs = dict(solver_kwargs)
        _max_retries = 10
        _seen_keys: set[str] = set()
        for _ in range(_max_retries):
            try:
                return self._path_solver(self._scene, **kwargs)
            except TypeError as exc:
                msg = str(exc)
                marker = "got an unexpected keyword argument '"
                if marker not in msg:
                    raise
                bad_key = msg.split(marker, 1)[1].split("'", 1)[0]
                if bad_key not in kwargs or bad_key in _seen_keys:
                    raise
                _seen_keys.add(bad_key)
                kwargs.pop(bad_key, None)
                self._warn_once(
                    f"unsupported::{bad_key}",
                    f"[SionnaAdapter] WARN: PathSolver does not support '{bad_key}'. Falling back without it.",
                )
        raise RuntimeError(
            f"PathSolver fallback exhausted after {_max_retries} retries; "
            f"dropped keys: {_seen_keys}"
        )

    def _extract_path_polylines(self, paths: object) -> list[dict]:
        """Extract receiver-reaching path polylines for debug visualization."""
        try:
            vertices = paths.vertices.numpy()
            valid = paths.valid.numpy()
            interactions = paths.interactions.numpy()
            src_positions = paths.sources.numpy().T
            tgt_positions = paths.targets.numpy().T
        except Exception as exc:
            self._warn_once("path_extract_base", f"[SionnaAdapter] WARN: path extraction failed: {exc}")
            return []

        max_depth = int(vertices.shape[0])
        num_paths = int(vertices.shape[-2]) if vertices.ndim >= 2 else 0
        if num_paths <= 0:
            return []

        num_src = int(src_positions.shape[0]) if src_positions.ndim >= 2 else 0
        num_tgt = int(tgt_positions.shape[0]) if tgt_positions.ndim >= 2 else 0

        try:
            if not bool(getattr(paths, "synthetic_array", True)):
                num_rx = int(paths.num_rx)
                rx_array_size = int(paths.rx_array.array_size)
                num_rx_patterns = int(len(paths.rx_array.antenna_pattern.patterns))
                num_tx = int(paths.num_tx)
                tx_array_size = int(paths.tx_array.array_size)
                num_tx_patterns = int(len(paths.tx_array.antenna_pattern.patterns))

                vertices = np.reshape(
                    vertices,
                    [max_depth, num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns, tx_array_size, -1, 3],
                )
                valid = np.reshape(valid, [num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns, tx_array_size, -1])
                interactions = np.reshape(
                    interactions,
                    [max_depth, num_rx, num_rx_patterns, rx_array_size, num_tx, num_tx_patterns, tx_array_size, -1],
                )

                vertices = vertices[:, :, 0, :, :, 0, :, :, :]
                valid = valid[:, 0, :, :, 0, :, :]
                interactions = interactions[:, :, 0, :, :, 0, :, :]
                vertices = np.reshape(vertices, [max_depth, num_tgt, num_src, -1, 3])
                valid = np.reshape(valid, [num_tgt, num_src, -1])
                interactions = np.reshape(interactions, [max_depth, num_tgt, num_src, -1])
        except Exception as exc:
            self._warn_once("path_extract_reshape", f"[SionnaAdapter] WARN: path extraction reshape failed: {exc}")
            return []

        out: list[dict] = []
        max_out = max(1, int(self._max_paths_to_draw))
        for rx in range(num_tgt):
            for tx in range(num_src):
                for p in range(num_paths):
                    if len(out) >= max_out:
                        return out
                    try:
                        if not bool(valid[rx, tx, p]):
                            continue
                    except Exception:
                        continue

                    src = [float(v) for v in src_positions[tx]]
                    tgt = [float(v) for v in tgt_positions[rx]]
                    points = [src]
                    interaction_types: list[int] = []
                    is_los = True

                    for i in range(max_depth):
                        t = int(interactions[i, rx, tx, p])
                        if t == 0:
                            break
                        is_los = False
                        v = vertices[i, rx, tx, p]
                        points.append([float(v[0]), float(v[1]), float(v[2])])
                        interaction_types.append(t)
                    points.append(tgt)
                    out.append(
                        {
                            "points_xyz": points,
                            "is_los": bool(is_los),
                            "interaction_types": interaction_types,
                            "num_segments": max(0, len(points) - 1),
                        }
                    )
        return out

    def get_last_path_geometry(self) -> list[dict]:
        """Return extracted path polylines from the latest snapshot call."""
        return list(self._last_path_geometry)

    def _cir_to_ofdm_explicit(self, a_flat: np.ndarray, tau_flat: np.ndarray) -> np.ndarray:
        """Compute OFDM CSI from CIR using explicit per-subcarrier phase terms."""
        if a_flat.size == 0 or tau_flat.size == 0:
            return np.zeros((self.num_subcarriers,), dtype=np.complex128)
        # Validate delays: clip negative values, warn on extreme delays
        neg_mask = tau_flat < 0
        if np.any(neg_mask):
            self._warn_once("tau_negative", "[SionnaAdapter] WARN: negative delay values clipped to 0")
            tau_flat = np.clip(tau_flat, 0, None)
        extreme_mask = tau_flat > 1e-3  # > 1ms
        if np.any(extreme_mask):
            self._warn_once("tau_extreme", "[SionnaAdapter] WARN: delay values > 1ms detected; possible numerical overflow")
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
            self._last_path_geometry = []
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
        if self._visualize_rays:
            self._last_path_geometry = self._extract_path_polylines(paths)
            if self._log_path_stats and self._last_path_geometry:
                seg_counts = [int(max(0, len(p.get("points_xyz", [])) - 1)) for p in self._last_path_geometry]
                los_count = sum(1 for p in self._last_path_geometry if bool(p.get("is_los", False)))
                nlos_count = max(0, len(self._last_path_geometry) - los_count)
                logger.info(
                    "ray_viz paths=%d los=%d nlos=%d seg_max=%d seg_mean=%.2f",
                    len(self._last_path_geometry), los_count, nlos_count,
                    max(seg_counts), float(np.mean(seg_counts)),
                )
        else:
            self._last_path_geometry = []

        a, tau = paths.cir(out_type="numpy")
        a_flat = np.asarray(a).reshape(-1)
        tau_flat = np.asarray(tau).reshape(-1)

        if a_flat.size == 0 or tau_flat.size == 0:
            a_flat = np.zeros((0,), dtype=np.complex128)
            tau_flat = np.zeros((0,), dtype=np.float64)

        if self.csi_method == "sionna_cfr":
            h = paths.cfr(self._frequencies, out_type="numpy")
            h_arr = np.asarray(h)
            if h_arr.ndim >= 1 and h_arr.shape[0] > 0:
                csi = h_arr.reshape(-1, h_arr.shape[-1])[0]
            elif h_arr.ndim >= 1:
                csi = np.zeros((self.num_subcarriers,), dtype=np.complex128)
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
