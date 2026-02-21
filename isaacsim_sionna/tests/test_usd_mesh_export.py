"""Unit tests for USD mesh extraction helper utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from isaacsim_sionna.bridge.usd_mesh_export import (
    UsdMeshPrimitive,
    export_meshes_to_ply,
    triangulate_faces,
)


def test_triangulate_faces_triangle_and_quad() -> None:
    face_counts = np.array([3, 4], dtype=np.int32)
    face_indices = np.array([0, 1, 2, 0, 2, 3, 4], dtype=np.int32)
    tris = triangulate_faces(face_counts=face_counts, face_indices=face_indices)

    assert tris.shape == (3, 3)
    assert tris.tolist() == [[0, 1, 2], [0, 2, 3], [0, 3, 4]]


def test_export_meshes_to_ply(tmp_path: Path) -> None:
    mesh = UsdMeshPrimitive(
        prim_path="/Root/TestMesh",
        vertices_xyz=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
    )

    refs = export_meshes_to_ply([mesh], output_dir=tmp_path)

    assert len(refs) == 1
    out_path = Path(refs[0].file_path)
    assert out_path.exists()

    text = out_path.read_text(encoding="utf-8")
    assert "element vertex 3" in text
    assert "element face 1" in text
    assert "3 0 1 2" in text
