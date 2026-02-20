#!/usr/bin/env python3
"""Pipeline runner for Isaac Sim -> Sionna CSI dataset collection."""

from __future__ import annotations

import argparse
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isaacsim_sionna.bridge.pipeline import Pipeline  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CSI pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--max-frames", type=int, default=None, help="Override runtime.max_frames")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.max_frames is not None:
        config.setdefault("runtime", {})["max_frames"] = int(args.max_frames)

    pipeline = Pipeline(config=config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
