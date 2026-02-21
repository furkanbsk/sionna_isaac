#!/usr/bin/env python3
"""
Debug script to visualize USD coordinate system by rendering from 10 camera positions.

This helps understand the spatial layout of the USD scene and align coordinates
between Isaac Sim and Sionna RT.

Usage:
    python isaacsim_sionna/scripts/debug_camera_positions.py \
        --scene isaacsim_sionna/data/scenes_usd/hospital.usd \
        --output isaacsim_sionna/data/debug_cameras

    # Or with custom config
    python isaacsim_sionna/scripts/debug_camera_positions.py \
        --config isaacsim_sionna/configs/hospital_static.yaml \
        --output isaacsim_sionna/data/debug_cameras
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Any

import yaml

# Camera positions: [position_xyz, target_xyz, description]
# Designed to cover different perspectives of the scene
CAMERA_POSITIONS = [
    # 1. Top-down center view (kuş bakışı - merkez)
    {
        "position": [0.0, 0.0, 20.0],
        "target": [0.0, 0.0, 0.0],
        "description": "Top-down center (20m yukarıdan)",
    },
    # 2. Top-down corner view (kuş bakışı - köşe)
    {
        "position": [10.0, 10.0, 15.0],
        "target": [0.0, 0.0, 0.0],
        "description": "Top-down from corner (10,10,15)",
    },
    # 3. Ground level - looking along X axis (zemin seviyesi - X ekseni)
    {
        "position": [0.0, -10.0, 1.5],
        "target": [0.0, 0.0, 1.5],
        "description": "Ground level, looking +X (insan boyu)",
    },
    # 4. Ground level - looking along Y axis (zemin seviyesi - Y ekseni)
    {
        "position": [-10.0, 0.0, 1.5],
        "target": [0.0, 0.0, 1.5],
        "description": "Ground level, looking +Y (insan boyu)",
    },
    # 5. Elevated corner view (yüksek köşe görünümü)
    {
        "position": [8.0, 8.0, 8.0],
        "target": [0.0, 0.0, 0.0],
        "description": "Elevated corner diagonal (8,8,8)",
    },
    # 6. Side view from +X (X ekseninden yan görünüm)
    {
        "position": [15.0, 0.0, 5.0],
        "target": [0.0, 0.0, 2.5],
        "description": "Side view from +X axis (15,0,5)",
    },
    # 7. Side view from -X (X eksi ekseninden yan görünüm)
    {
        "position": [-15.0, 0.0, 5.0],
        "target": [0.0, 0.0, 2.5],
        "description": "Side view from -X axis (-15,0,5)",
    },
    # 8. Side view from +Y (Y ekseninden yan görünüm)
    {
        "position": [0.0, 15.0, 5.0],
        "target": [0.0, 0.0, 2.5],
        "description": "Side view from +Y axis (0,15,5)",
    },
    # 9. Side view from -Y (Y eksi ekseninden yan görünüm)
    {
        "position": [0.0, -15.0, 5.0],
        "target": [0.0, 0.0, 2.5],
        "description": "Side view from -Y axis (0,-15,5)",
    },
    # 10. Isometric view (izometrik görünüm)
    {
        "position": [10.0, -10.0, 10.0],
        "target": [0.0, 0.0, 0.0],
        "description": "Isometric view (10,-10,10)",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug: Render scene from 10 camera positions")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Path to YAML config (optional, scene_usd will be read from it)",
    )
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        default=None,
        help="Path to USD scene file (overrides config if provided)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("isaacsim_sionna/data/debug_cameras"),
        help="Output directory for rendered images",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution (default: 1280 720)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image format (default: png)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (default: True)",
    )
    parser.add_argument(
        "--no-headless",
        action="store_false",
        dest="headless",
        help="Show Isaac Sim GUI (for debugging)",
    )
    return parser.parse_args()


def save_image(path: pathlib.Path, rgb: Any, image_format: str) -> None:
    """Save RGB array as image file."""
    import numpy as np

    arr = np.asarray(rgb)
    if arr.ndim != 3:
        raise ValueError(f"Expected RGB image rank 3, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    try:
        from PIL import Image

        img = Image.fromarray(arr, mode="RGB")
        if image_format in {"jpg", "jpeg"}:
            img.save(str(path), format="JPEG", quality=95)
        else:
            img.save(str(path), format="PNG")
        return
    except Exception:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(str(path), arr)
        return
    except Exception as exc:
        raise RuntimeError("No image writer backend available (PIL/imageio)") from exc


def compute_bbox_from_meshes(mesh_aabbs: list[dict]) -> dict[str, list[float]] | None:
    """Compute global bounding box from mesh AABBs."""
    if not mesh_aabbs:
        return None

    all_min = [float("inf")] * 3
    all_max = [float("-inf")] * 3

    for mesh in mesh_aabbs:
        center = mesh.get("center_xyz", [0, 0, 0])
        half_extent = mesh.get("half_extent_xyz", [0, 0, 0])
        for i in range(3):
            all_min[i] = min(all_min[i], center[i] - half_extent[i])
            all_max[i] = max(all_max[i], center[i] + half_extent[i])

    return {"min_xyz": all_min, "max_xyz": all_max}


def extract_mesh_aabbs_from_usd(scene_path: pathlib.Path, max_meshes: int = 256) -> list[dict]:
    """Extract mesh AABBs from USD file for bbox estimation."""
    from isaacsim_sionna.bridge.usd_to_sionna import extract_mesh_aabbs_from_usd_file

    try:
        aabbs = extract_mesh_aabbs_from_usd_file(str(scene_path), max_meshes=max_meshes)
        return [
            {
                "prim_path": m.prim_path,
                "center_xyz": m.center_xyz,
                "half_extent_xyz": m.half_extent_xyz,
            }
            for m in aabbs
        ]
    except Exception as exc:
        print(f"[WARN] Failed to extract mesh AABBs: {exc}")
        return []


def main() -> int:
    args = parse_args()

    # Determine scene path
    scene_path = args.scene
    if scene_path is None and args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        scene_path = pathlib.Path(config.get("scenario", {}).get("scene_usd", ""))

    if scene_path is None or not scene_path.exists():
        print(f"ERROR: USD scene file not found: {scene_path}")
        return 1

    print(f"[DEBUG] Scene: {scene_path}")
    print(f"[DEBUG] Output: {args.output}")
    print(f"[DEBUG] Resolution: {args.resolution}")
    print(f"[DEBUG] Format: {args.format}")

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    renders_dir = args.output / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Extract scene bbox for adaptive camera positioning
    print("[DEBUG] Extracting scene bounding box...")
    mesh_aabbs = extract_mesh_aabbs_from_usd(scene_path, max_meshes=256)
    scene_bbox = compute_bbox_from_meshes(mesh_aabbs)

    if scene_bbox:
        min_xyz = scene_bbox["min_xyz"]
        max_xyz = scene_bbox["max_xyz"]
        center = [(min_xyz[i] + max_xyz[i]) / 2 for i in range(3)]
        size = [max_xyz[i] - min_xyz[i] for i in range(3)]
        print(f"[DEBUG] Scene BBOX: min={min_xyz}, max={max_xyz}")
        print(f"[DEBUG] Scene Center: {center}")
        print(f"[DEBUG] Scene Size: {size}")

        # Adapt camera positions to scene bbox
        diagonal = (size[0] ** 2 + size[1] ** 2 + size[2] ** 2) ** 0.5
        print(f"[DEBUG] Scene Diagonal: {diagonal:.2f}m")
    else:
        center = [0.0, 0.0, 0.0]
        diagonal = 20.0
        print("[DEBUG] Using default bbox (no mesh data extracted)")

    # Start Isaac Sim
    print("[DEBUG] Starting Isaac Sim...")
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": args.headless})

    # Import Isaac APIs after SimulationApp bootstrap
    import omni.usd
    from isaacsim.core.utils.stage import open_stage, update_stage

    # Open the USD stage
    print(f"[DEBUG] Opening stage: {scene_path}")
    ctx = omni.usd.get_context()
    if not open_stage(str(scene_path.resolve())):
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

    # Initialize Replicator
    print("[DEBUG] Initializing Replicator...")
    import omni.replicator.core as rep

    # Create metadata file
    metadata = {
        "scene_path": str(scene_path),
        "scene_bbox": scene_bbox,
        "scene_center": center,
        "scene_diagonal_m": diagonal,
        "mesh_count": len(mesh_aabbs),
        "timestamp": datetime.now().isoformat(),
        "camera_positions": [],
    }

    # Render from each camera position sequentially
    print(f"[DEBUG] Rendering {len(CAMERA_POSITIONS)} camera views...\n")

    for i, cam_info in enumerate(CAMERA_POSITIONS):
        pos = cam_info["position"]
        target = cam_info["target"]
        desc = cam_info["description"]

        print(f"[{i+1:02d}/{len(CAMERA_POSITIONS)}] {desc}")
        print(f"       Position: {pos}")
        print(f"       Target:   {target}")

        # Create camera and render product
        camera = rep.create.camera(position=tuple(pos), look_at=tuple(target))
        render_product = rep.create.render_product(camera, tuple(args.resolution))
        
        # Attach annotator
        rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annotator.attach(render_product)

        # Flush render and capture
        rep.orchestrator.step()
        rgb = rgb_annotator.get_data()

        if rgb is not None:
            # Save image
            img_path = renders_dir / f"cam_{i+1:02d}_{desc.split()[0]}.png"
            save_image(img_path, rgb, "png")
            print(f"       Saved: {img_path.relative_to(args.output)}")

            # Record metadata
            metadata["camera_positions"].append(
                {
                    "index": i + 1,
                    "position_xyz": pos,
                    "target_xyz": target,
                    "description": desc,
                    "image_path": str(img_path.relative_to(args.output)),
                    "resolution": args.resolution,
                }
            )
        else:
            print(f"       [WARN] Failed to capture frame")

        # Cleanup for this camera
        try:
            rgb_annotator.detach(render_product)
        except Exception:
            pass
        try:
            render_product.destroy()
        except Exception:
            pass
        try:
            camera.destroy()
        except Exception:
            pass

    # Write metadata
    metadata_path = args.output / "camera_positions.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[DEBUG] Metadata saved: {metadata_path}")

    # Create index HTML for easy viewing
    html_path = args.output / "index.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Camera Position Debug - {scene_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #76b900; }}
        .info {{ background: #2a2a2a; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
        .card img {{ width: 100%; height: auto; display: block; }}
        .card-info {{ padding: 15px; }}
        .card-info h3 {{ margin: 0 0 10px 0; color: #76b900; }}
        .card-info p {{ margin: 5px 0; font-size: 13px; }}
        .coord {{ font-family: monospace; background: #333; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>🎥 Camera Position Debug</h1>
    <div class="info">
        <h2>Scene: {scene_path.name}</h2>
        <p><strong>Path:</strong> {scene_path}</p>
        <p><strong>Mesh Count:</strong> {len(mesh_aabbs)}</p>
        <p><strong>Timestamp:</strong> {metadata["timestamp"]}</p>
"""

    if scene_bbox:
        html_content += f"""
        <p><strong>BBOX Min:</strong> <span class="coord">{scene_bbox["min_xyz"]}</span></p>
        <p><strong>BBOX Max:</strong> <span class="coord">{scene_bbox["max_xyz"]}</span></p>
        <p><strong>Center:</strong> <span class="coord">{center}</span></p>
        <p><strong>Size:</strong> <span class="coord">{size}</span></p>
        <p><strong>Diagonal:</strong> {diagonal:.2f}m</p>
"""

    html_content += """
    </div>
    <h2>Camera Views</h2>
    <div class="grid">
"""

    for cam in metadata["camera_positions"]:
        html_content += f"""
        <div class="card">
            <img src="{cam["image_path"]}" alt="Camera {cam["index"]}">
            <div class="card-info">
                <h3>#{cam["index"]} - {cam["description"]}</h3>
                <p><strong>Position:</strong> <span class="coord">{cam["position_xyz"]}</span></p>
                <p><strong>Target:</strong> <span class="coord">{cam["target_xyz"]}</span></p>
                <p><strong>Resolution:</strong> {cam["resolution"][0]}x{cam["resolution"][1]}</p>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    with html_path.open("w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[DEBUG] HTML index created: {html_path}")

    # Cleanup
    sim_app.close()
    print("\n[DEBUG] Done!")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
