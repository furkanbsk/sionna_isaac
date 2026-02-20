"""Contract tests for SionnaAdapter config validation paths."""

from __future__ import annotations

import pytest

from isaacsim_sionna.bridge.sionna_adapter import SionnaAdapter


def test_initialize_requires_geometry_proxy() -> None:
    adapter = SionnaAdapter({"radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64}})
    with pytest.raises(ValueError, match="geometry_proxy is required"):
        adapter.initialize(None)


def test_initialize_rejects_empty_mesh_proxy() -> None:
    adapter = SionnaAdapter({"radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64}})
    with pytest.raises(RuntimeError, match="No mesh AABBs"):
        adapter.initialize({"mesh_aabbs": []})
