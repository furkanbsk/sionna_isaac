"""Sionna RT adapter with runtime USD geometry proxy and PathSolver."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sionna.rt import (
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
    subcarrier_frequencies,
)

from isaacsim_sionna.bridge.usd_to_sionna import MeshAabb, build_sionna_xml_from_aabbs


class SionnaAdapter:
    """Channel engine using Sionna RT PathSolver with runtime scene XML."""

    def __init__(self, config: dict):
        self.config = config
        self.radio_cfg = config.get("radio", {})
        self.runtime_cfg = config.get("runtime", {})
        self.scenario_cfg = config.get("scenario", {})

        self.carrier_hz = float(self.radio_cfg.get("carrier_hz", 3.5e9))
        self.bandwidth_hz = float(self.radio_cfg.get("bandwidth_hz", 20e6))
        self.num_subcarriers = int(self.radio_cfg.get("num_subcarriers", 256))
        self.max_depth = int(self.radio_cfg.get("max_depth", 5))
        self.samples_per_src = int(float(self.radio_cfg.get("samples_per_src", 100000)))

        self._state: dict | None = None
        self._scene = None
        self._path_solver = PathSolver()
        self._frequencies = None
        self._mesh_count = 0

    def initialize(self, geometry_proxy: dict | None = None) -> None:
        """Build runtime scene XML from geometry proxy and initialize solvers."""
        if self.num_subcarriers < 1:
            raise ValueError("radio.num_subcarriers must be >= 1")
        if self.bandwidth_hz <= 0.0:
            raise ValueError("radio.bandwidth_hz must be > 0")
        if self.carrier_hz <= 0.0:
            raise ValueError("radio.carrier_hz must be > 0")

        if geometry_proxy is None:
            raise ValueError("geometry_proxy is required for SionnaAdapter.initialize")

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

        cache_xml = self.scenario_cfg.get(
            "scene_sionna_cache",
            "isaacsim_sionna/data/scenes_sionna/cache/runtime_scene.xml",
        )
        cache_xml_path = Path(cache_xml)
        build_sionna_xml_from_aabbs(mesh_aabbs, cache_xml_path)

        self._scene = load_scene(str(cache_xml_path.resolve()), merge_shapes=False)

        # Apply per-mesh transforms from extracted USD AABBs.
        for i, mesh in enumerate(mesh_aabbs):
            obj = self._scene.get(f"mesh_box_{i}")
            obj.position = mesh.center_xyz
            obj.scaling = mesh.half_extent_xyz

        self._scene.frequency = float(self.carrier_hz)

        # Single-antenna baseline. This can be upgraded later via config.
        array = PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="V",
        )
        self._scene.tx_array = array
        self._scene.rx_array = array

        tx = Transmitter(name="tx", position=[0.0, 0.0, 1.5])
        rx = Receiver(name="rx", position=[1.0, 0.0, 1.5])
        self._scene.add(tx)
        self._scene.add(rx)

        subcarrier_spacing = self.bandwidth_hz / self.num_subcarriers
        self._frequencies = subcarrier_frequencies(self.num_subcarriers, subcarrier_spacing)
        self._mesh_count = len(mesh_aabbs)

        print(
            "[SionnaAdapter] initialized "
            f"carrier_hz={self.carrier_hz:.3e} "
            f"bandwidth_hz={self.bandwidth_hz:.3e} "
            f"num_subcarriers={self.num_subcarriers} "
            f"mesh_boxes={self._mesh_count}"
        )

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

    def compute_snapshot(self) -> dict:
        """Return one JSON-serializable CIR and CSI snapshot from PathSolver."""
        if self._scene is None or self._frequencies is None:
            raise RuntimeError("SionnaAdapter not initialized. Call initialize() first.")

        tx_pose = (self._state or {}).get("tx_pose")
        rx_pose = (self._state or {}).get("rx_pose")
        if tx_pose is None or rx_pose is None:
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
            }

        paths = self._path_solver(
            self._scene,
            max_depth=self.max_depth,
            samples_per_src=self.samples_per_src,
            los=True,
            specular_reflection=True,
            diffuse_reflection=False,
            refraction=False,
            diffraction=False,
        )

        a, tau = paths.cir(out_type="numpy")
        h = paths.cfr(self._frequencies, out_type="numpy")

        # a,tau shape for 1x1 setup is low-dimensional; flatten for JSON.
        a_flat = np.asarray(a).reshape(-1)
        tau_flat = np.asarray(tau).reshape(-1)

        h_arr = np.asarray(h)
        # Flatten all but subcarrier axis and take first stream for baseline serialization.
        if h_arr.ndim >= 1:
            csi = h_arr.reshape(-1, h_arr.shape[-1])[0]
        else:
            csi = h_arr

        tx = np.asarray(tx_pose["pos_xyz"], dtype=np.float64)
        rx = np.asarray(rx_pose["pos_xyz"], dtype=np.float64)
        distance_m = float(np.linalg.norm(rx - tx))

        return {
            "status": "ok",
            "a_re": np.real(a_flat).astype(np.float64).tolist(),
            "a_im": np.imag(a_flat).astype(np.float64).tolist(),
            "tau_s": np.real(tau_flat).astype(np.float64).tolist(),
            "csi_re": np.real(csi).astype(np.float64).tolist(),
            "csi_im": np.imag(csi).astype(np.float64).tolist(),
            "distance_m": distance_m,
            "num_paths": int(a_flat.size),
            "path_types_present": ["los", "specular"],
            "mesh_boxes": self._mesh_count,
        }
