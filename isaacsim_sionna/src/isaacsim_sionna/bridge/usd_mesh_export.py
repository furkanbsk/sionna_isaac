"""Extract USD mesh geometry and export runtime mesh files for Sionna."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass
class UsdMeshPrimitive:
    """Triangulated world-space mesh primitive extracted from USD."""

    prim_path: str
    vertices_xyz: np.ndarray
    triangles: np.ndarray


@dataclass
class MeshFileRef:
    """Reference to one exported mesh file."""

    prim_path: str
    file_path: str


def triangulate_faces(face_counts: np.ndarray, face_indices: np.ndarray) -> np.ndarray:
    """Triangulate polygonal faces using fan triangulation."""
    expected_len = int(face_counts.sum())
    if len(face_indices) < expected_len:
        raise ValueError(
            f"face_indices length ({len(face_indices)}) < sum(face_counts) ({expected_len})"
        )
    tris: list[list[int]] = []
    cursor = 0
    for n in face_counts.tolist():
        if n < 3:
            cursor += n
            continue
        face = face_indices[cursor : cursor + n]
        cursor += n
        anchor = int(face[0])
        for i in range(1, n - 1):
            tris.append([anchor, int(face[i]), int(face[i + 1])])
    if not tris:
        return np.zeros((0, 3), dtype=np.int32)
    return np.asarray(tris, dtype=np.int32)


_COMPILED_REGEX_CACHE: dict[str, re.Pattern] = {}


def _compile_regex(pattern: str) -> re.Pattern:
    """Compile and cache regex patterns."""
    if pattern not in _COMPILED_REGEX_CACHE:
        try:
            _COMPILED_REGEX_CACHE[pattern] = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid mesh filter regex '{pattern}': {exc}") from exc
    return _COMPILED_REGEX_CACHE[pattern]


def _matches_filters(prim_path: str, include_regex: str | None, exclude_regex: str | None) -> bool:
    if include_regex and _compile_regex(include_regex).search(prim_path) is None:
        return False
    if exclude_regex and _compile_regex(exclude_regex).search(prim_path) is not None:
        return False
    return True


def extract_mesh_primitives(
    scene_usd: str,
    max_meshes: int | None = None,
    include_regex: str | None = None,
    exclude_regex: str | None = None,
) -> list[UsdMeshPrimitive]:
    """Extract world-space triangulated meshes from USD file."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(Path(scene_usd).resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {scene_usd}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    extracted: list[UsdMeshPrimitive] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        prim_path = str(prim.GetPath())
        if not _matches_filters(prim_path, include_regex=include_regex, exclude_regex=exclude_regex):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()

        if points is None or face_counts is None or face_indices is None:
            continue

        local_vertices = np.asarray([[float(p[0]), float(p[1]), float(p[2])] for p in points], dtype=np.float64)
        if local_vertices.size == 0:
            continue

        world_xf = xform_cache.GetLocalToWorldTransform(prim)
        world_vertices = []
        for v in local_vertices:
            wp = world_xf.Transform(Gf.Vec3d(float(v[0]), float(v[1]), float(v[2])))
            world_vertices.append([float(wp[0]), float(wp[1]), float(wp[2])])

        triangles = triangulate_faces(
            face_counts=np.asarray(face_counts, dtype=np.int32),
            face_indices=np.asarray(face_indices, dtype=np.int32),
        )
        if triangles.shape[0] == 0:
            continue

        extracted.append(
            UsdMeshPrimitive(
                prim_path=prim_path,
                vertices_xyz=np.asarray(world_vertices, dtype=np.float64),
                triangles=triangles,
            )
        )

        if max_meshes is not None and len(extracted) >= max_meshes:
            break

    return extracted


def _safe_name_from_prim_path(prim_path: str) -> str:
    return prim_path.strip("/").replace("/", "_").replace(":", "_")


def _write_ascii_ply(mesh: UsdMeshPrimitive, out_path: Path) -> None:
    verts = mesh.vertices_xyz
    tris = mesh.triangles

    lines: list[str] = []
    lines.append("ply")
    lines.append("format ascii 1.0")
    lines.append(f"element vertex {verts.shape[0]}")
    lines.append("property float x")
    lines.append("property float y")
    lines.append("property float z")
    lines.append(f"element face {tris.shape[0]}")
    lines.append("property list uchar int vertex_indices")
    lines.append("end_header")

    for v in verts:
        lines.append(f"{v[0]:.9f} {v[1]:.9f} {v[2]:.9f}")

    for tri in tris:
        lines.append(f"3 {int(tri[0])} {int(tri[1])} {int(tri[2])}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_meshes_to_ply(meshes: list[UsdMeshPrimitive], output_dir: str | Path) -> list[MeshFileRef]:
    """Export mesh list to ASCII PLY files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    refs: list[MeshFileRef] = []
    for i, mesh in enumerate(meshes):
        fname = f"mesh_{i:04d}_{_safe_name_from_prim_path(mesh.prim_path)}.ply"
        out_path = out_dir / fname
        _write_ascii_ply(mesh, out_path)
        refs.append(MeshFileRef(prim_path=mesh.prim_path, file_path=str(out_path.resolve())))

    return refs
