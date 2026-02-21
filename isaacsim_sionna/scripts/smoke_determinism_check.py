#!/usr/bin/env python3
"""Run pipeline twice and compare output hashes."""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import shutil
import sys
import traceback

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from isaacsim_sionna.bridge.pipeline import Pipeline  # pylint: disable=wrong-import-position


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Determinism smoke check")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--workdir", default="/tmp/isaacsim_sionna_determinism", help="Temporary output root")
    p.add_argument("--max-frames", type=int, default=1, help="Frames for each run")
    return p.parse_args()


def _run_once(base_cfg: dict, out_root: pathlib.Path, max_frames: int) -> dict:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("runtime", {})["max_frames"] = int(max_frames)
    cfg.setdefault("project", {})["output_root"] = str(out_root)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    try:
        Pipeline(cfg).run()
    except Exception as exc:  # pragma: no cover - runtime diagnostics
        print(f"[smoke_determinism] run failure at {out_root}: {exc!r}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise

    manifest_path = out_root / "manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Expected manifest not found: {manifest_path}")
    samples_path = out_root / "samples.jsonl"
    if not samples_path.exists():
        raise RuntimeError(f"Expected samples not found: {samples_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return {
        "manifest": manifest,
        "rows": rows,
    }


def _manifest_without_timestamp(manifest: dict) -> dict:
    out = copy.deepcopy(manifest)
    out.pop("run_timestamp_utc", None)
    return out


def _extract_path_signature(rows: list[dict]) -> list[dict]:
    signatures = []
    for row in rows:
        snap = row.get("snapshot", {})
        signatures.append(
            {
                "frame_idx": row.get("frame_idx"),
                "num_paths": snap.get("num_paths"),
                "a_re": snap.get("a_re"),
                "a_im": snap.get("a_im"),
                "tau_s": snap.get("tau_s"),
            }
        )
    return signatures


def main() -> int:
    args = parse_args()
    base_cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    workdir = pathlib.Path(args.workdir)
    run1 = _run_once(base_cfg, workdir / "run1", args.max_frames)
    run2 = _run_once(base_cfg, workdir / "run2", args.max_frames)

    man1 = run1["manifest"]
    man2 = run2["manifest"]
    out1 = man1.get("outputs", {})
    out2 = man2.get("outputs", {})
    sig1 = _extract_path_signature(run1["rows"])
    sig2 = _extract_path_signature(run2["rows"])

    print("same_manifest_except_timestamp:", _manifest_without_timestamp(man1) == _manifest_without_timestamp(man2))
    print("run1.samples_sha256:", out1.get("samples_sha256"))
    print("run2.samples_sha256:", out2.get("samples_sha256"))
    print("same_samples_hash:", out1.get("samples_sha256") == out2.get("samples_sha256"))
    print("same_num_samples:", out1.get("num_samples") == out2.get("num_samples"))
    print("same_path_signatures:", sig1 == sig2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
