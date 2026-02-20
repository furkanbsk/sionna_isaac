"""Runtime bridge from Isaac USD file to Sionna scene XML.

This bridge approximates each USD mesh by its world-axis-aligned box. It is a
fast, deterministic proxy that uses real scene geometry extents and enables
multipath with PathSolver.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


@dataclass
class MeshAabb:
    """World-space axis-aligned bounding box for one mesh prim."""

    prim_path: str
    center_xyz: list[float]
    half_extent_xyz: list[float]


def _collect_mesh_aabbs_from_stage(stage, max_meshes: int | None = None) -> list[MeshAabb]:
    from pxr import Usd, UsdGeom

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    aabbs: list[MeshAabb] = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        bbox = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        mn = bbox.GetMin()
        mx = bbox.GetMax()

        min_xyz = [float(mn[0]), float(mn[1]), float(mn[2])]
        max_xyz = [float(mx[0]), float(mx[1]), float(mx[2])]

        if not all(math.isfinite(v) for v in min_xyz + max_xyz):
            continue

        size_xyz = [max_xyz[i] - min_xyz[i] for i in range(3)]
        # Keep thin surfaces (walls/floors); only skip fully degenerate meshes.
        if max(size_xyz) <= 1e-5:
            continue

        center_xyz = [(min_xyz[i] + max_xyz[i]) * 0.5 for i in range(3)]
        half_extent_xyz = [max(size_xyz[i] * 0.5, 0.01) for i in range(3)]

        aabbs.append(
            MeshAabb(
                prim_path=str(prim.GetPath()),
                center_xyz=center_xyz,
                half_extent_xyz=half_extent_xyz,
            )
        )

        if max_meshes is not None and len(aabbs) >= max_meshes:
            break

    return aabbs


def extract_mesh_aabbs_from_usd_file(scene_usd: str, max_meshes: int | None = None) -> list[MeshAabb]:
    """Extract mesh AABBs from a USD file using pxr Stage.Open."""
    from pxr import Usd

    stage = Usd.Stage.Open(str(Path(scene_usd).resolve()))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {scene_usd}")
    return _collect_mesh_aabbs_from_stage(stage, max_meshes=max_meshes)


def extract_mesh_aabbs_from_open_stage(max_meshes: int | None = None) -> list[MeshAabb]:
    """Extract mesh AABBs from currently opened Isaac USD stage."""
    import omni.usd  # Imported lazily; requires SimulationApp runtime.

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is currently open in Isaac context")
    return _collect_mesh_aabbs_from_stage(stage, max_meshes=max_meshes)


def compute_global_bbox(aabbs: list[MeshAabb]) -> dict[str, list[float]]:
    """Compute world bbox that encloses all mesh AABBs."""
    if not aabbs:
        raise ValueError("Cannot compute global bbox: empty mesh list")

    min_xyz = [float("inf"), float("inf"), float("inf")]
    max_xyz = [float("-inf"), float("-inf"), float("-inf")]

    for mesh in aabbs:
        for i in range(3):
            mn = mesh.center_xyz[i] - mesh.half_extent_xyz[i]
            mx = mesh.center_xyz[i] + mesh.half_extent_xyz[i]
            min_xyz[i] = min(min_xyz[i], mn)
            max_xyz[i] = max(max_xyz[i], mx)

    return {"min_xyz": min_xyz, "max_xyz": max_xyz}


def build_sionna_xml_from_aabbs(aabbs: list[MeshAabb], output_xml: Path) -> Path:
    """Write a Mitsuba/Sionna XML scene with one cube per mesh AABB."""
    if not aabbs:
        raise ValueError("Cannot build Sionna XML: no mesh AABBs")

    output_xml = Path(output_xml)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append('<scene version="2.1.0">')
    lines.append('  <bsdf type="itu-radio-material" id="mat-default">')
    lines.append('    <string name="type" value="concrete"/>')
    lines.append('    <float name="thickness" value="0.2"/>')
    lines.append('  </bsdf>')

    for i, _ in enumerate(aabbs):
        lines.append(f'  <shape type="cube" id="mesh_box_{i}">')
        lines.append('    <ref name="bsdf" id="mat-default"/>')
        lines.append('  </shape>')

    lines.append('</scene>')
    output_xml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_xml
