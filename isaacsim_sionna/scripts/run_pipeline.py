#!/usr/bin/env python3
"""Pipeline runner for Isaac Sim -> Sionna CSI dataset collection."""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

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
    parser.add_argument(
        "--validate-after-run",
        action="store_true",
        help="Run post-run dataset QA checks and fail if QA fails",
    )
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
    try:
        pipeline.run()
        if args.validate_after_run:
            from isaacsim_sionna.qa.dataset_validator import (  # pylint: disable=import-outside-toplevel
                validate_run,
            )

            run_root = pathlib.Path(config.get("project", {}).get("output_root", "isaacsim_sionna/data/raw"))
            strict = bool((config.get("qa") or {}).get("strict", True))
            report = validate_run(run_root=run_root, config=config, strict=strict, write_manifest=True)
            print(
                f"[run_pipeline] QA status={report['status']} "
                f"failures={report['summary']['num_failures']} "
                f"run_root={run_root}"
            )
            if int(report["exit_code"]) != 0:
                return int(report["exit_code"])
        return 0
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"[run_pipeline] ERROR: {exc!r}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
