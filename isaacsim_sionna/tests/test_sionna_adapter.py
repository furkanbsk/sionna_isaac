"""Contract tests for SionnaAdapter config validation paths."""

from __future__ import annotations

import numpy as np
import pytest

from isaacsim_sionna.bridge.sionna_adapter import SionnaAdapter


def _base_config() -> dict:
    return {
        "radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64},
        "isaac": {"geometry": {"mode": "aabb"}},
    }


def test_initialize_requires_geometry_proxy() -> None:
    adapter = SionnaAdapter(_base_config())
    with pytest.raises(ValueError, match="geometry_proxy is required"):
        adapter.initialize(None)


def test_initialize_rejects_empty_aabb_proxy() -> None:
    adapter = SionnaAdapter(_base_config())
    with pytest.raises(RuntimeError, match="No mesh AABBs"):
        adapter.initialize({"geometry_mode": "aabb", "mesh_aabbs": []})


def test_initialize_requires_scene_usd_for_mesh_mode() -> None:
    cfg = {
        "radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64},
        "isaac": {"geometry": {"mode": "mesh"}},
    }
    adapter = SionnaAdapter(cfg)
    with pytest.raises(RuntimeError, match="requires scene_usd"):
        adapter.initialize({"geometry_mode": "mesh"})


def test_resolve_solver_seed_frame_offset_strategy() -> None:
    cfg = {
        "project": {"seed": 100, "solver_seed_strategy": "frame_offset"},
        "radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64},
        "isaac": {"geometry": {"mode": "aabb"}},
    }
    adapter = SionnaAdapter(cfg)
    assert adapter._resolve_solver_seed(frame_idx=0) == 100  # pylint: disable=protected-access
    assert adapter._resolve_solver_seed(frame_idx=5) == 105  # pylint: disable=protected-access


def test_resolve_solver_seed_fixed_strategy() -> None:
    cfg = {
        "project": {"seed": 100, "solver_seed_strategy": "fixed"},
        "radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64},
        "isaac": {"geometry": {"mode": "aabb"}},
    }
    adapter = SionnaAdapter(cfg)
    assert adapter._resolve_solver_seed(frame_idx=0) == 100  # pylint: disable=protected-access
    assert adapter._resolve_solver_seed(frame_idx=5) == 100  # pylint: disable=protected-access


def test_update_dynamic_state_updates_actor_proxy_positions() -> None:
    class _Obj:
        def __init__(self):
            self.position = None
            self.orientation = None

    class _Scene:
        def __init__(self):
            self._objects = {"tx": _Obj(), "rx": _Obj(), "actor_proxy_0": _Obj()}

        def get(self, key):
            return self._objects[key]

    adapter = SionnaAdapter(_base_config())
    adapter._scene = _Scene()  # pylint: disable=protected-access
    adapter._dynamic_actor_proxy_map = {"/World/humanoid_01": "actor_proxy_0"}  # pylint: disable=protected-access

    state = {
        "tx_pose": {"pos_xyz": [1.0, 2.0, 3.0]},
        "rx_pose": {"pos_xyz": [4.0, 5.0, 6.0]},
        "actor_poses": [
            {
                "prim_path": "/World/humanoid_01",
                "position_xyz": [7.0, 8.0, 9.0],
                "orientation_quat_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        ],
    }

    adapter.update_dynamic_state(state)

    assert adapter._scene.get("tx").position == [1.0, 2.0, 3.0]  # pylint: disable=protected-access
    assert adapter._scene.get("rx").position == [4.0, 5.0, 6.0]  # pylint: disable=protected-access
    assert adapter._scene.get("actor_proxy_0").position == [7.0, 8.0, 9.0]  # pylint: disable=protected-access


def test_propagation_flags_default_to_rich_model() -> None:
    adapter = SionnaAdapter(_base_config())
    assert adapter.propagation_flags["los"] is True
    assert adapter.propagation_flags["specular_reflection"] is True
    assert adapter.propagation_flags["diffuse_reflection"] is True
    assert adapter.propagation_flags["refraction"] is True
    assert adapter.propagation_flags["diffraction"] is True
    assert adapter.propagation_flags["edge_diffraction"] is True
    assert adapter.propagation_flags["diffraction_lit_region"] is True


def test_compute_paths_with_fallback_drops_unsupported_kwargs() -> None:
    adapter = SionnaAdapter(_base_config())
    calls = {"n": 0}

    class _Paths:
        pass

    def _solver(scene, **kwargs):
        _ = scene
        calls["n"] += 1
        if "edge_diffraction" in kwargs:
            raise TypeError("foo got an unexpected keyword argument 'edge_diffraction'")
        return _Paths()

    adapter._scene = object()  # pylint: disable=protected-access
    adapter._path_solver = _solver  # pylint: disable=protected-access
    out = adapter._compute_paths_with_fallback(  # pylint: disable=protected-access
        {"max_depth": 5, "edge_diffraction": True, "los": True}
    )
    assert isinstance(out, _Paths)
    assert calls["n"] == 2


def test_explicit_cir_to_ofdm_is_frequency_selective() -> None:
    cfg = {
        "radio": {"carrier_hz": 5.32e9, "bandwidth_hz": 40e6, "num_subcarriers": 64},
        "isaac": {"geometry": {"mode": "aabb"}},
    }
    adapter = SionnaAdapter(cfg)
    a = np.asarray([1.0 + 0j, 0.8 - 0.2j], dtype=np.complex128)
    tau = np.asarray([0.0, 35e-9], dtype=np.float64)
    h = adapter._cir_to_ofdm_explicit(a, tau)  # pylint: disable=protected-access
    assert h.shape == (64,)
    # Ensure per-subcarrier magnitude is not flat
    assert float(np.std(np.abs(h))) > 1e-3
