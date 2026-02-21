#!/usr/bin/env python3
"""
Debug script to explore hospital USD scene structure and find TX/RX/actor positions.

This helps understand:
1. Scene bounding box
2. Existing TX/RX prim locations
3. Actor prim paths
4. Room/corridor layout

Usage:
    python isaacsim_sionna/scripts/debug_scene_structure.py \
        --scene isaacsim_sionna/data/scenes_usd/hospital.usd
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug: Explore USD scene structure")
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        required=True,
        help="Path to USD scene file",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("isaacsim_sionna/data/debug_scene"),
        help="Output directory for debug info",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.scene.exists():
        print(f"ERROR: USD scene file not found: {args.scene}")
        return 1

    print(f"[DEBUG] Scene: {args.scene}")
    print(f"[DEBUG] Output: {args.output}")

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Start Isaac Sim
    print("[DEBUG] Starting Isaac Sim...")
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True})

    # Import Isaac APIs after SimulationApp bootstrap
    import omni.usd
    from isaacsim.core.utils.stage import open_stage, update_stage

    # Open the USD stage
    print(f"[DEBUG] Opening stage: {args.scene}")
    ctx = omni.usd.get_context()
    if not open_stage(str(args.scene.resolve())):
        print("[ERROR] Failed to open USD stage")
        sim_app.close()
        return 1
    update_stage()

    # Get stage info
    stage = ctx.get_stage()
    if stage is None:
        print("[ERROR] Failed to get USD stage")
        sim_app.close()
        return 1

    print(f"[DEBUG] Stage opened: {stage.GetRootLayer().identifier}")

    # Traverse the stage and collect prim info
    print("\n[DEBUG] Traversing USD stage...\n")

    prim_info = []
    scene_bbox_min = [float("inf")] * 3
    scene_bbox_max = [float("-inf")] * 3

    def traverse_prim(prim, depth=0):
        """Recursively traverse prims and collect info."""
        import pxr

        path = str(prim.GetPath())
        prim_type = prim.GetTypeName()
        
        # Get prim kind from UsdModelAPI
        try:
            from pxr import UsdModelAPI
            prim_kind = UsdModelAPI(prim).GetKind()
        except Exception:
            prim_kind = ""

        # Get transform
        try:
            from pxr import UsdGeom

            xform = UsdGeom.Xformable(prim)
            matrix = xform.GetLocalTransformation()
            pos = matrix[3][:3]  # Translation
        except Exception:
            pos = [0.0, 0.0, 0.0]

        # Update scene bbox
        for i in range(3):
            scene_bbox_min[i] = min(scene_bbox_min[i], pos[i])
            scene_bbox_max[i] = max(scene_bbox_max[i], pos[i])

        info = {
            "path": path,
            "type": prim_type,
            "kind": str(prim_kind),
            "position": [float(v) for v in pos],
            "depth": depth,
            "children_count": len(prim.GetChildren()),
        }
        prim_info.append(info)

        # Print interesting prims
        if depth <= 2 or prim_type or prim_kind != "":
            indent = "  " * depth
            marker = ""
            if "tx" in path.lower() or "rx" in path.lower() or "transmitter" in path.lower() or "receiver" in path.lower():
                marker = "  <<< TX/RX?"
            if "actor" in path.lower() or "human" in path.lower() or "man" in path.lower():
                marker = "  <<< ACTOR?"
            if prim_type in ["Xform", "Scope", "Geometry", "Mesh"]:
                marker = "  <<< GEOMETRY"

            print(f"{indent}[{prim_type or 'N/A':15s}] {path}{marker}")

        # Recurse into children (limit depth)
        if depth < 4:
            for child in prim.GetChildren():
                traverse_prim(child, depth + 1)

    # Start traversal from root
    root_prim = stage.GetPseudoRoot()
    traverse_prim(root_prim)

    # Print scene bbox
    print("\n" + "=" * 60)
    print("SCENE BOUNDING BOX:")
    print(f"  Min: {scene_bbox_min}")
    print(f"  Max: {scene_bbox_max}")

    scene_center = [(scene_bbox_min[i] + scene_bbox_max[i]) / 2 for i in range(3)]
    scene_size = [scene_bbox_max[i] - scene_bbox_min[i] for i in range(3)]
    print(f"  Center: {scene_center}")
    print(f"  Size: {scene_size}")

    diagonal = (scene_size[0] ** 2 + scene_size[1] ** 2 + scene_size[2] ** 2) ** 0.5
    print(f"  Diagonal: {diagonal:.2f}m")
    print("=" * 60)

    # Find TX/RX candidates
    print("\n[SEARCH] Looking for TX/RX/Actor prims...\n")

    tx_rx_candidates = []
    actor_candidates = []

    for info in prim_info:
        path_lower = info["path"].lower()
        if any(kw in path_lower for kw in ["tx", "transmitter", "receiver", "rx"]):
            tx_rx_candidates.append(info)
            print(f"  [TX/RX] {info['path']} @ {info['position']}")
        if any(kw in path_lower for kw in ["actor", "human", "man", "agent", "character"]):
            actor_candidates.append(info)
            print(f"  [ACTOR] {info['path']} @ {info['position']}")

    # Save metadata
    metadata = {
        "scene_path": str(args.scene),
        "scene_bbox": {
            "min_xyz": scene_bbox_min,
            "max_xyz": scene_bbox_max,
            "center_xyz": scene_center,
            "size_xyz": scene_size,
            "diagonal_m": diagonal,
        },
        "prim_count": len(prim_info),
        "tx_rx_candidates": tx_rx_candidates,
        "actor_candidates": actor_candidates,
        "all_prims": prim_info,
    }

    metadata_path = args.output / "scene_structure.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n[DEBUG] Metadata saved: {metadata_path}")

    # Suggest camera positions based on scene bbox
    print("\n" + "=" * 60)
    print("SUGGESTED CAMERA POSITIONS FOR DATA COLLECTION AREA:")
    print("=" * 60)

    # Calculate camera positions based on scene bbox
    margin = 2.0
    eye_height = 1.5  # Human eye level

    suggestions = [
        {
            "name": "Corner overview (high)",
            "position": [scene_bbox_min[0] - margin, scene_bbox_min[1] - margin, scene_bbox_max[2] + margin],
            "target": scene_center,
            "description": "Köşeden genel bakış",
        },
        {
            "name": "Ground level entrance",
            "position": [scene_bbox_min[0] - margin, scene_center[1], eye_height],
            "target": [scene_center[0], scene_center[1], eye_height],
            "description": "Zemin seviyesi giriş",
        },
        {
            "name": "Center corridor view",
            "position": [scene_center[0], scene_center[1] - scene_size[1] / 2 - margin, eye_height],
            "target": [scene_center[0], scene_center[1], eye_height],
            "description": "Merkez koridor görünümü",
        },
        {
            "name": "Opposite corner",
            "position": [scene_bbox_max[0] + margin, scene_bbox_max[1] + margin, eye_height],
            "target": scene_center,
            "description": "Karşı köşe",
        },
        {
            "name": "Top-down full view",
            "position": [scene_center[0], scene_center[1], scene_bbox_max[2] + margin * 2],
            "target": scene_center,
            "description": "Tam tepeden görünüm",
        },
    ]

    for i, sug in enumerate(suggestions, 1):
        print(f"\n{i}. {sug['name']}:")
        print(f"   Position: {sug['position']}")
        print(f"   Target:   {sug['target']}")
        print(f"   Desc: {sug['description']}")

    # Save suggested camera positions
    suggestions_path = args.output / "suggested_cameras.json"
    with suggestions_path.open("w", encoding="utf-8") as f:
        json.dump(suggestions, f, indent=2)
    print(f"\n[DEBUG] Suggested cameras saved: {suggestions_path}")

    # Cleanup
    sim_app.close()
    print("\n[DEBUG] Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
