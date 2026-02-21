#!/usr/bin/env python3
"""Validate generated dataset artifacts after pipeline run."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isaacsim_sionna.qa.dataset_validator import validate_run  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate generated CSI dataset artifacts")
    p.add_argument("--run-root", required=True, help="Directory containing manifest.json and samples.jsonl")
    p.add_argument("--config", default=None, help="Optional YAML config override")
    p.add_argument("--strict", action="store_true", default=True, help="Fail run on any failed check")
    p.add_argument("--no-strict", action="store_false", dest="strict", help="Do not fail exit code on QA issues")
    p.add_argument("--write-manifest", action="store_true", default=True, help="Write QA report into manifest")
    p.add_argument("--no-write-manifest", action="store_false", dest="write_manifest")
    p.add_argument("--print-report", action="store_true", help="Print full JSON QA report")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = None
    if args.config:
        cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))

    report = validate_run(
        run_root=args.run_root,
        config=cfg,
        strict=bool(args.strict),
        write_manifest=bool(args.write_manifest),
    )

    print(f"qa_status={report['status']}")
    print(f"qa_failures={report['summary']['num_failures']}")
    if args.print_report:
        print(json.dumps(report, indent=2, sort_keys=True))

    return int(report["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
