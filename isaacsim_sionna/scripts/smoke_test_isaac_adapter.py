#!/usr/bin/env python3
"""Smoke test for IsaacAdapter: boot + step + pose contract."""

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
    parser = argparse.ArgumentParser(description="Smoke test IsaacAdapter")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames to step")
    return parser.parse_args()


def _validate_pose_block(name: str, pose: dict | None) -> None:
    if pose is None:
        raise RuntimeError(f"Missing required pose block: {name}")
    for key in ["prim_path", "pos_xyz", "quat_wxyz"]:
        if key not in pose:
            raise RuntimeError(f"Pose block '{name}' missing key '{key}'")
    if len(pose["pos_xyz"]) != 3:
        raise RuntimeError(f"Pose block '{name}' pos_xyz length must be 3")
    if len(pose["quat_wxyz"]) != 4:
        raise RuntimeError(f"Pose block '{name}' quat_wxyz length must be 4")


def _validate_state(state: dict) -> None:
    required = [
        "timestamp_sim",
        "frame_idx",
        "tx_pose",
        "rx_pose",
        "actors",
        "stage_source",
        "is_playing",
    ]
    for k in required:
        if k not in state:
            raise RuntimeError(f"State missing required key: {k}")

    _validate_pose_block("tx_pose", state["tx_pose"])
    _validate_pose_block("rx_pose", state["rx_pose"])

    if not isinstance(state["actors"], list):
        raise RuntimeError("State 'actors' must be a list")


def main() -> int:
    args = parse_args()
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return 1

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    adapter = IsaacAdapter(config)

    try:
        adapter.start()

        last_state = None
        for _ in range(args.frames):
            adapter.step()
            state = adapter.get_state()
            _validate_state(state)
            last_state = state

        assert last_state is not None
        print("[SmokeTest] PASS")
        print(
            "[SmokeTest] summary "
            f"stage_source={last_state['stage_source']} "
            f"frame_idx={last_state['frame_idx']} "
            f"timestamp_sim={last_state['timestamp_sim']:.6f}"
        )
        return 0
    except Exception as exc:
        print(f"[SmokeTest] FAIL: {exc}")
        return 2
    finally:
        adapter.stop()


if __name__ == "__main__":
    raise SystemExit(main())
