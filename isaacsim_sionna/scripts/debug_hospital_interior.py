#!/usr/bin/env python3
"""
Debug script to visualize hospital interior from 10 positions.

Shows:
1. TX/RX anchor positions (based on bbox auto-anchor)
2. Interior viewpoints at human eye level (1.5m)
3. Different rooms and corridors

Usage:
    python isaacsim_sionna/scripts/debug_hospital_interior.py \
        --scene isaacsim_sionna/data/scenes_usd/hospital.usd \
        --output isaacsim_sionna/data/debug_hospital
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime
from typing import Any

import yaml

# Camera positions designed for hospital interior visualization
# Isaac Sim rotation order: [roll, pitch, yaw] in degrees (X, Y, Z rotation)
# Z-up coordinate system
CAMERA_POSITIONS = [
    # 1. Center looking +Y (default, no rotation)
    {
        "position": [0.0, 0.0, 1.5],
        "rotation": [0.0, 0.0, 0.0],  # roll=0, pitch=0, yaw=0
        "description": "Merkez default",
    },
    # 2. Center looking -Y (rotate 180 around Z axis = yaw)
    {
        "position": [0.0, 0.0, 1.5],
        "rotation": [0.0, 0.0, 180.0],  # yaw=180
        "description": "Merkez Z-rot 180",
    },
    # 3. Center looking +X (rotate 90 around Z axis = yaw)
    {
        "position": [0.0, 0.0, 1.5],
        "rotation": [0.0, 0.0, 90.0],  # yaw=90
        "description": "Merkez Z-rot 90",
    },
    # 4. Center looking -X (rotate -90 around Z axis = yaw)
    {
        "position": [0.0, 0.0, 1.5],
        "rotation": [0.0, 0.0, -90.0],  # yaw=-90
        "description": "Merkez Z-rot -90",
    },
    # 5. Corner (-1,-1) looking diagonal (yaw=45) - MOVED CLOSER TO CENTER
    {
        "position": [-1.0, -1.0, 1.5],
        "rotation": [0.0, 0.0, 45.0],  # yaw=45
        "description": "Köşe (-1,-1) Z-rot 45",
    },
    # 6. Corner (3,3) looking diagonal (yaw=225)
    {
        "position": [3.0, 3.0, 1.5],
        "rotation": [0.0, 0.0, 225.0],  # yaw=225
        "description": "Köşe (3,3) Z-rot 225",
    },
    # 7. Corner (-3,3) looking diagonal (yaw=-45)
    {
        "position": [-3.0, 3.0, 1.5],
        "rotation": [0.0, 0.0, -45.0],  # yaw=-45
        "description": "Köşe (-3,3) Z-rot -45",
    },
    # 8. Corner (1,-1) looking diagonal (yaw=135) - MOVED CLOSER TO CENTER
    {
        "position": [1.0, -1.0, 1.5],
        "rotation": [0.0, 0.0, 135.0],  # yaw=135
        "description": "Köşe (1,-1) Z-rot 135",
    },
    # 9. Edge (0,-3) looking +Y (default) - MOVED CLOSER
    {
        "position": [0.0, -3.0, 1.5],
        "rotation": [0.0, 0.0, 0.0],  # default
        "description": "Kenar (0,-3) default",
    },
    # 10. Top-down: LOWER height, rotate -90 around X axis to look down
    {
        "position": [0.0, 0.0, 3.0],  # LOWER from 4.0 to 3.0
        "rotation": [-90.0, 0.0, 0.0],  # roll=-90 (tilt down)
        "description": "Tepeden X-rot -90 düşük",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug: Hospital interior camera views")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=None,
        help="Path to YAML config (optional)",
    )
    parser.add_argument(
        "--scene",
        type=pathlib.Path,
        default=None,
        help="Path to USD scene file (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("isaacsim_sionna/data/debug_hospital"),
        help="Output directory",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[1280, 720],
        metavar=("WIDTH", "HEIGHT"),
        help="Image resolution",
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
        img.save(str(path), format="PNG")
        return
    except Exception:
        pass

    try:
        import imageio.v2 as imageio
        imageio.imwrite(str(path), arr)
        return
    except Exception as exc:
        raise RuntimeError("No image writer backend available") from exc


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

    # Setup output directory
    args.output.mkdir(parents=True, exist_ok=True)
    renders_dir = args.output / "renders"
    renders_dir.mkdir(parents=True, exist_ok=True)

    # Start Isaac Sim
    print("[DEBUG] Starting Isaac Sim...")
    from isaacsim import SimulationApp

    sim_app = SimulationApp({"headless": True})

    # Import Isaac APIs after SimulationApp bootstrap
    import omni.usd
    from isaacsim.core.utils.stage import open_stage, update_stage
    import omni.replicator.core as rep

    # Open the USD stage
    print(f"[DEBUG] Opening stage: {scene_path}")
    ctx = omni.usd.get_context()
    if not open_stage(str(scene_path.resolve())):
        print("[ERROR] Failed to open USD stage")
        sim_app.close()
        return 1
    update_stage()

    stage = ctx.get_context().get_stage() if hasattr(ctx, 'get_context') else ctx.get_stage()
    if stage is None:
        stage = omni.usd.get_context().get_stage()
    
    if stage is None:
        print("[ERROR] Failed to get stage")
        sim_app.close()
        return 1

    print(f"[DEBUG] Stage opened: {stage.GetRootLayer().identifier}")

    # Metadata
    metadata = {
        "scene_path": str(scene_path),
        "timestamp": datetime.now().isoformat(),
        "camera_positions": [],
    }

    # Render from each camera position
    print(f"\n[DEBUG] Rendering {len(CAMERA_POSITIONS)} interior views...\n")

    for i, cam_info in enumerate(CAMERA_POSITIONS):
        pos = cam_info["position"]
        rot = cam_info.get("rotation", [0.0, 0.0, 0.0])
        desc = cam_info["description"]

        print(f"[{i+1:02d}/{len(CAMERA_POSITIONS)}] {desc}")
        print(f"       Position: {pos}")
        print(f"       Rotation: {rot} (roll, pitch, yaw)")

        # Create camera with explicit rotation (Euler angles)
        camera = rep.create.camera(
            position=tuple(pos),
            rotation=tuple(rot)
        )
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
            print(f"       ✓ Saved: {img_path.relative_to(args.output)}")

            # Record metadata
            metadata["camera_positions"].append({
                "index": i + 1,
                "position_xyz": pos,
                "rotation_deg": rot,
                "description": desc,
                "image_path": str(img_path.relative_to(args.output)),
                "resolution": args.resolution,
            })
        else:
            print(f"       ✗ [WARN] Failed to capture frame")

        # Cleanup
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

    # Create HTML index
    html_path = args.output / "index.html"
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Hospital Interior Debug - {scene_path.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #eee; }}
        h1 {{ color: #76b900; }}
        .info {{ background: #2a2a2a; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 20px; }}
        .card {{ background: #2a2a2a; border-radius: 8px; overflow: hidden; }}
        .card img {{ width: 100%; height: auto; display: block; }}
        .card-info {{ padding: 15px; }}
        .card-info h3 {{ margin: 0 0 10px 0; color: #76b900; }}
        .card-info p {{ margin: 5px 0; font-size: 13px; }}
        .coord {{ font-family: monospace; background: #333; padding: 2px 6px; border-radius: 3px; }}
        .note {{ background: #3a3a2a; padding: 10px; border-left: 3px solid #76b900; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>🏥 Hospital Interior Camera Debug</h1>
    <div class="info">
        <h2>Scene: {scene_path.name}</h2>
        <p><strong>Path:</strong> {scene_path}</p>
        <p><strong>Timestamp:</strong> {metadata["timestamp"]}</p>
        <div class="note">
            <strong>📌 Note:</strong> All cameras at human eye level (1.5m) with proper up vector [0,0,1].
            TX anchor at ~20% of bbox, RX anchor at ~80% of bbox.
        </div>
    </div>
    <h2>Camera Views ({len(metadata["camera_positions"])})</h2>
    <div class="grid">
"""

    for cam in metadata["camera_positions"]:
        html_content += f"""
        <div class="card">
            <img src="{cam["image_path"]}" alt="Camera {cam["index"]}">
            <div class="card-info">
                <h3>#{cam["index"]} - {cam["description"]}</h3>
                <p><strong>Position:</strong> <span class="coord">{cam["position_xyz"]}</span></p>
                <p><strong>Rotation:</strong> <span class="coord">{cam["rotation_deg"]}</span> (roll, pitch, yaw)</p>
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
