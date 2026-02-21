#!/usr/bin/env python3
"""Compare one-frame pipeline outputs for aabb vs mesh geometry modes."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isaacsim_sionna.bridge.pipeline import Pipeline  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare aabb and mesh geometry modes")
    p.add_argument("--config", required=True, help="Base YAML config")
    p.add_argument("--workdir", default="/tmp/isaacsim_sionna_compare", help="Temp output dir")
    return p.parse_args()


def _run_once(config: dict, mode: str, workdir: pathlib.Path) -> dict:
    cfg = copy.deepcopy(config)
    cfg.setdefault("runtime", {})["max_frames"] = 1
    cfg.setdefault("isaac", {}).setdefault("geometry", {})["mode"] = mode

    out_root = workdir / mode / "raw"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("project", {})["output_root"] = str(out_root)

    pipeline = Pipeline(cfg)
    pipeline.run()

    rows = [
        json.loads(line)
        for line in (out_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"No samples generated for mode={mode}")

    snap = rows[-1]["snapshot"]
    csi_re = np.asarray(snap.get("csi_re", []), dtype=np.float64)
    return {
        "mode": mode,
        "num_paths": int(snap.get("num_paths", 0) or 0),
        "geometry_mode": snap.get("geometry_mode"),
        "mesh_boxes": snap.get("mesh_boxes"),
        "mesh_file_count": snap.get("mesh_file_count"),
        "csi_re_std": float(csi_re.std()) if csi_re.size > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    cfg_path = pathlib.Path(args.config)
    workdir = pathlib.Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    aabb = _run_once(cfg, mode="aabb", workdir=workdir)
    mesh = _run_once(cfg, mode="mesh", workdir=workdir)

    print("AABB:", aabb)
    print("MESH:", mesh)
    print("CHECK num_paths(mesh)>=num_paths(aabb):", mesh["num_paths"] >= aabb["num_paths"])
    print("CHECK csi_std(mesh)>=0:", mesh["csi_re_std"] >= 0.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
