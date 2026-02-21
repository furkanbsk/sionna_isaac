"""Unit tests for runtime USD->Sionna bridge helpers."""

from __future__ import annotations

from pathlib import Path

from isaacsim_sionna.bridge.usd_mesh_export import MeshFileRef
from isaacsim_sionna.bridge.usd_to_sionna import (
    MeshAabb,
    build_sionna_xml_from_aabbs,
    build_sionna_xml_from_mesh_files,
    compute_global_bbox,
)


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
    out = tmp_path / "scene_aabb.xml"
    build_sionna_xml_from_aabbs(aabbs, out)
    text = out.read_text(encoding="utf-8")
    assert '<scene version="2.1.0">' in text
    assert '<shape type="cube" id="mesh_box_0">' in text


def test_build_sionna_xml_from_mesh_files(tmp_path: Path) -> None:
    mesh_ply = tmp_path / "mesh_0000_test.ply"
    mesh_ply.write_text("ply\nformat ascii 1.0\nend_header\n", encoding="utf-8")

    out = tmp_path / "scene_mesh.xml"
    refs = [MeshFileRef(prim_path="/Root/Test", file_path=str(mesh_ply.resolve()))]
    build_sionna_xml_from_mesh_files(refs, out)

    text = out.read_text(encoding="utf-8")
    assert '<shape type="ply" id="mesh_0">' in text
    assert str(mesh_ply.resolve()) in text
