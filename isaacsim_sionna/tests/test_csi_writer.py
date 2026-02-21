from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from isaacsim_sionna.exporters.csi_writer import CsiWriter


def _run_context() -> dict:
    return {
        "timestamp_utc": "2026-02-20T00:00:00Z",
        "project_name": "demo",
        "seed": 42,
        "hash_algo": "sha256",
        "config_hash": "abc123",
        "git": {"available": True, "commit": "deadbeef", "is_dirty": False},
        "runtime": {"seeded_libraries": ["random", "numpy"]},
        "config": {"project": {"seed": 42}},
    }


def test_writer_emits_manifest_and_stable_output_hash(tmp_path: Path) -> None:
    cfg = {"project": {"output_root": str(tmp_path / "out")}}
    writer = CsiWriter(cfg)
    writer.open(run_context=_run_context())
    writer.write(
        frame_idx=0,
        state={"timestamp_sim": 0.0, "actor_poses": [{"prim_path": "/World/a0"}]},
        snapshot={"status": "ok", "csi_re": [1.0], "timestamp_sim": 0.0},
    )
    writer.write(
        frame_idx=1,
        state={"timestamp_sim": 1.0, "actor_poses": [{"prim_path": "/World/a0"}]},
        snapshot={"status": "ok", "csi_re": [2.0], "timestamp_sim": 1.0},
    )
    writer.close()

    out_root = Path(cfg["project"]["output_root"])
    samples_path = out_root / "samples.jsonl"
    manifest_path = out_root / "manifest.json"
    assert samples_path.exists()
    assert manifest_path.exists()

    expected_hash = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row0 = json.loads(samples_path.read_text(encoding="utf-8").splitlines()[0])
    assert row0["timestamp_sim"] == 0.0
    assert row0["actor_poses"][0]["prim_path"] == "/World/a0"
    assert row0["image_path"] is None
    assert manifest["outputs"]["num_samples"] == 2
    assert manifest["outputs"]["samples_sha256"] == expected_hash


def test_activity_label_schedule_applied_per_frame(tmp_path: Path) -> None:
    cfg = {
        "project": {"output_root": str(tmp_path / "out")},
        "labels": {
            "activity_label": "idle",
            "activity_schedule": [
                {"start_frame": 0, "end_frame": 2, "label": "walking"},
                {"start_frame": 3, "end_frame": 5, "label": "falling"},
            ],
        },
    }
    writer = CsiWriter(cfg)
    writer.open(run_context=_run_context())
    writer.write(frame_idx=1, state={"timestamp_sim": 0.1}, snapshot={"status": "ok", "csi_re": [1.0], "csi_im": [0.0]})
    writer.write(frame_idx=4, state={"timestamp_sim": 0.2}, snapshot={"status": "ok", "csi_re": [1.0], "csi_im": [0.0]})
    writer.write(frame_idx=8, state={"timestamp_sim": 0.3}, snapshot={"status": "ok", "csi_re": [1.0], "csi_im": [0.0]})
    writer.close()

    rows = [
        json.loads(line)
        for line in (Path(cfg["project"]["output_root"]) / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[0]["activity_label"] == "walking"
    assert rows[1]["activity_label"] == "falling"
    assert rows[2]["activity_label"] == "idle"
    assert rows[0]["image_path"] is None


@pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py not installed")
def test_writer_tensor_mode_uses_jsonl_as_index_ledger(tmp_path: Path) -> None:
    cfg = {
        "project": {"output_root": str(tmp_path / "out")},
        "labels": {"activity_label": "walking"},
        "storage": {
            "tensor_store": {
                "enabled": True,
                "path": "csi_tensors.h5",
                "dtype": "complex64",
                "compression": None,
                "chunk_frames": 8,
            }
        },
    }
    writer = CsiWriter(cfg)
    writer.open(run_context=_run_context())
    writer.write(
        frame_idx=3,
        state={"timestamp_sim": 0.3, "actor_poses": [{"prim_path": "/World/h0"}]},
        snapshot={
            "status": "ok",
            "num_paths": 2,
            "csi_re": [1.0, 2.0],
            "csi_im": [0.5, -0.5],
            "a_re": [0.1],
            "a_im": [0.2],
            "tau_s": [1e-9],
        },
    )
    writer.close()

    out_root = Path(cfg["project"]["output_root"])
    row = json.loads((out_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))

    assert row["activity_label"] == "walking"
    assert "tensor_ref" in row
    assert "csi_re" not in row["snapshot"]
    assert "csi_im" not in row["snapshot"]
    assert manifest["outputs"]["tensor_store_file"] == "csi_tensors.h5"
    assert manifest["outputs"]["tensor_store_rows"] == 1
    assert manifest["outputs"]["tensor_store_sha256"] is not None


def test_writer_logs_image_path_when_render_ref_exists(tmp_path: Path) -> None:
    cfg = {
        "project": {"output_root": str(tmp_path / "out")},
        "isaac": {"camera": {"enabled": True, "output_subdir": "renders"}},
    }
    writer = CsiWriter(cfg)
    writer.open(run_context=_run_context())
    writer.write(
        frame_idx=0,
        state={"timestamp_sim": 0.0, "actor_poses": []},
        snapshot={"status": "ok", "csi_re": [1.0], "csi_im": [0.0]},
        render_ref={"file": "renders/frame_0000.png", "width": 1280, "height": 720},
    )
    writer.close()

    out_root = Path(cfg["project"]["output_root"])
    row = json.loads((out_root / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))
    assert row["image_path"] == "renders/frame_0000.png"
    assert row["render_ref"]["file"] == "renders/frame_0000.png"
    assert manifest["outputs"]["num_rendered_frames"] == 1
