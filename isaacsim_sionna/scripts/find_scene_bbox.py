#!/usr/bin/env python3
"""
Simple script to find hospital scene bounding box and suggest camera positions.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find scene bounding box")
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        required=True,
        help="Path to USD scene file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.scene.exists():
        print(f"ERROR: USD scene file not found: {args.scene}")
        return 1

    print(f"[DEBUG] Scene: {args.scene}")

    # Start Isaac Sim
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": True})

    # Import after SimulationApp
    import omni.usd
    from isaacsim.core.utils.stage import open_stage, update_stage
    from pxr import UsdGeom

    # Open stage
    ctx = omni.usd.get_context()
    if not open_stage(str(args.scene.resolve())):
        print("[ERROR] Failed to open USD stage")
        sim_app.close()
        return 1
    update_stage()

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[ERROR] Failed to get stage")
        sim_app.close()
        return 1

    print(f"[DEBUG] Stage opened: {stage.GetRootLayer().identifier}")

    # Compute bounding box of entire scene
    print("\n[DEBUG] Computing scene bounding box...\n")

    bbox_min = [float("inf")] * 3
    bbox_max = [float("-inf")] * 3
    prim_count = 0

    def process_prim(prim, depth=0):
        nonlocal prim_count
        import pxr

        # Try to get bounds
        try:
            xform = UsdGeom.Xformable(prim)
            bound = xform.GetLocalBound()
            if bound:
                range3d = bound.GetRange()
                if not range3d.IsEmpty():
                    min_pt = range3d.GetMin()
                    max_pt = range3d.GetMax()
                    for i in range(3):
                        bbox_min[i] = min(bbox_min[i], float(min_pt[i]))
                        bbox_max[i] = max(bbox_max[i], float(max_pt[i]))
                    prim_count += 1

                    # Print first few prims with bounds
                    if prim_count <= 10:
                        path = str(prim.GetPath())
                        print(f"  [{prim_count:3d}] {path:60s} min={list(min_pt)} max={list(max_pt)}")
        except Exception:
            pass

        # Recurse (limit depth)
        if depth < 3:
            for child in prim.GetChildren():
                process_prim(child, depth + 1)

    # Start from pseudo-root
    root = stage.GetPseudoRoot()
    process_prim(root)

    print(f"\n[DEBUG] Processed {prim_count} prims with bounds")

    # Print scene bbox
    print("\n" + "=" * 70)
    print("SCENE BOUNDING BOX:")
    print(f"  Min XYZ: {bbox_min}")
    print(f"  Max XYZ: {bbox_max}")

    if bbox_min[0] != float("inf"):
        center = [(bbox_min[i] + bbox_max[i]) / 2 for i in range(3)]
        size = [bbox_max[i] - bbox_min[i] for i in range(3)]
        diagonal = (size[0]**2 + size[1]**2 + size[2]**2) ** 0.5

        print(f"  Center:  {center}")
        print(f"  Size:    {size}")
        print(f"  Diagonal: {diagonal:.2f}m")

        # Suggested camera positions
        print("\n" + "=" * 70)
        print("SUGGESTED CAMERA POSITIONS (at 1.5m eye level):")
        print("=" * 70)

        margin = 1.0
        eye_height = 1.5

        suggestions = [
            ("Center looking +Y", [center[0], center[1], eye_height], [center[0], bbox_max[1], eye_height]),
            ("Center looking -Y", [center[0], center[1], eye_height], [center[0], bbox_min[1], eye_height]),
            ("Center looking +X", [center[0], center[1], eye_height], [bbox_max[0], center[1], eye_height]),
            ("Center looking -X", [center[0], center[1], eye_height], [bbox_min[0], center[1], eye_height]),
            ("Corner min,min", [bbox_min[0]+margin, bbox_min[1]+margin, eye_height], center),
            ("Corner max,max", [bbox_max[0]-margin, bbox_max[1]-margin, eye_height], center),
            ("Corner min,max", [bbox_min[0]+margin, bbox_max[1]-margin, eye_height], center),
            ("Corner max,min", [bbox_max[0]-margin, bbox_min[1]+margin, eye_height], center),
            ("Top-down center", [center[0], center[1], bbox_max[2]+2], center),
            ("TX anchor (20%)", [
                bbox_min[0] + 0.2 * size[0],
                bbox_min[1] + 0.2 * size[1],
                eye_height
            ], center),
            ("RX anchor (80%)", [
                bbox_min[0] + 0.8 * size[0],
                bbox_min[1] + 0.8 * size[1],
                eye_height
            ], center),
        ]

        for i, (name, pos, target) in enumerate(suggestions, 1):
            print(f"\n{i:2d}. {name}:")
            print(f"    position={pos}")
            print(f"    target={target}")

    print("=" * 70)

    # Cleanup
    sim_app.close()
    print("\n[DEBUG] Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
