"""Unit tests for runtime USD->Sionna bridge helpers."""

from __future__ import annotations

from pathlib import Path

from isaacsim_sionna.bridge.usd_to_sionna import MeshAabb, build_sionna_xml_from_aabbs, compute_global_bbox


def test_compute_global_bbox() -> None:
    aabbs = [
        MeshAabb("/A", center_xyz=[0.0, 0.0, 1.0], half_extent_xyz=[1.0, 2.0, 0.5]),
        MeshAabb("/B", center_xyz=[5.0, -1.0, 0.0], half_extent_xyz=[0.5, 0.5, 1.0]),
    ]
    bbox = compute_global_bbox(aabbs)
    assert bbox["min_xyz"] == [-1.0, -2.0, -1.0]
    assert bbox["max_xyz"] == [5.5, 2.0, 1.5]


def test_build_sionna_xml_from_aabbs(tmp_path: Path) -> None:
    aabbs = [MeshAabb("/M", center_xyz=[0.0, 0.0, 0.0], half_extent_xyz=[1.0, 1.0, 1.0])]
    out = tmp_path / "scene.xml"
    build_sionna_xml_from_aabbs(aabbs, out)
    text = out.read_text(encoding="utf-8")
    assert '<scene version="2.1.0">' in text
    assert '<shape type="cube" id="mesh_box_0">' in text
