#!/usr/bin/env python3
"""Smoke test: initialize Isaac camera and capture one RGB frame."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isaacsim_sionna.bridge.isaac_adapter import IsaacAdapter  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test camera initialization/capture")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--frame-idx",
        type=int,
        default=0,
        help="Frame index tag for saved image filename",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config file not found: {cfg_path}")
        return 1

    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config.setdefault("isaac", {}).setdefault("camera", {})["enabled"] = True

    isaac = IsaacAdapter(config=config)
    try:
        isaac.start()
        isaac.step()
        ref = isaac.capture_rgb(frame_idx=int(args.frame_idx))
        if ref is None:
            print("camera_smoke=failed reason=no_render_ref")
            return 1
        print("camera_smoke=passed")
        print(f"render_file={ref.get('file')}")
        print(f"camera_name={ref.get('camera_name')}")
        print(f"resolution={ref.get('width')}x{ref.get('height')}")
        return 0
    finally:
        isaac.stop()


if __name__ == "__main__":
    raise SystemExit(main())
