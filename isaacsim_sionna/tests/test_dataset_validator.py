from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from isaacsim_sionna.qa.dataset_validator import validate_run


pytestmark = pytest.mark.skipif(importlib.util.find_spec("h5py") is None, reason="h5py not installed")


def _write_artifacts(
    tmp_path: Path,
    rows: list[dict],
    csi: np.ndarray,
    frame_idx: list[int],
    config: dict,
    renders_dir: str | None = None,
) -> Path:
    import h5py

    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "samples.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8"
    )

    with h5py.File(str(run_root / "csi_tensors.h5"), "w") as h5:
        frames = h5.create_group("frames")
        frames.create_dataset("csi_c64", data=csi.astype(np.complex64))
        frames.create_dataset("frame_idx", data=np.asarray(frame_idx, dtype=np.int32))
        frames.create_dataset("num_paths", data=np.asarray([3] * len(frame_idx), dtype=np.int32))

    manifest = {
        "run_timestamp_utc": "2026-02-21T00:00:00Z",
        "project_name": "qa_test",
        "seed": 42,
        "config_hash": "abc",
        "hash_algo": "sha256",
        "outputs": {
            "samples_jsonl": "samples.jsonl",
            "num_samples": len(rows),
            "samples_sha256": "x",
            "tensor_store_file": "csi_tensors.h5",
            "tensor_store_rows": len(rows),
            "tensor_store_sha256": "y",
        },
        "config": config,
    }
    if renders_dir is not None:
        manifest["outputs"]["renders_dir"] = str(renders_dir)
    (run_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return run_root


def _base_rows(n: int) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append(
            {
                "frame_idx": i,
                "timestamp_sim": float(i) * 0.1,
                "activity_label": "walking",
                "tensor_ref": {"file": "csi_tensors.h5", "group": "/frames", "row": i, "dataset": "csi_c64"},
                "snapshot": {"num_paths": 3, "status": "ok"},
                "state": {
                    "tx_pose": {"pos_xyz": [0.0, 0.0, 1.5]},
                    "rx_pose": {"pos_xyz": [5.0, 0.0, 1.5]},
                },
            }
        )
    return rows


def test_validate_run_pass(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {
            "path_count": {"max_los_only_ratio": 0.8},
            "csi_variance": {"adaptive_scale": 1e-4, "abs_floor": 1e-8},
            "bounds": {"margin_m": 0.1},
            "strict": True,
        },
    }
    rows = _base_rows(3)
    csi = np.asarray(
        [
            [1.0 + 0.0j, 2.0 + 0.0j],
            [1.01 + 0.01j, 2.01 + 0.01j],
            [1.03 + 0.02j, 2.03 + 0.02j],
        ],
        dtype=np.complex64,
    )

    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1, 2], config)

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=True)
    assert report["status"] == "passed"
    assert report["exit_code"] == 0

    manifest = json.loads((run_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["qa_status"] == "passed"
    assert manifest["qa"]["summary"]["num_failures"] == 0


def test_validate_run_fails_on_stale_csi(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {"strict": True},
    }
    rows = _base_rows(3)
    csi = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [1.0 + 0.0j, 2.0 + 0.0j], [1.0 + 0.0j, 2.0 + 0.0j]], dtype=np.complex64)
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1, 2], config)

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    assert report["status"] == "failed"
    assert report["exit_code"] == 1
    failed_names = {item["name"] for item in report["results"] if item["status"] == "failed"}
    assert "csi_variance_floor" in failed_names


def test_validate_run_fails_sync_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {"strict": True},
    }
    rows = _base_rows(3)
    rows[1]["tensor_ref"]["row"] = 99
    csi = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [1.1 + 0.1j, 2.1 + 0.1j], [1.2 + 0.2j, 2.2 + 0.2j]], dtype=np.complex64)
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1, 2], config)

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    assert report["status"] == "failed"
    failed_names = {item["name"] for item in report["results"] if item["status"] == "failed"}
    assert "sync_check" in failed_names


def test_validate_run_fails_frequency_selectivity(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {
            "strict": True,
            "frequency_selectivity": {"metric": "p90", "abs_floor": 1e-4},
        },
    }
    rows = _base_rows(4)
    # Constant over subcarriers in every frame -> physically flat channel.
    csi = np.asarray(
        [
            [1.0 + 0.0j, 1.0 + 0.0j],
            [1.1 + 0.1j, 1.1 + 0.1j],
            [1.2 + 0.2j, 1.2 + 0.2j],
            [1.3 + 0.3j, 1.3 + 0.3j],
        ],
        dtype=np.complex64,
    )
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1, 2, 3], config)

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    assert report["status"] == "failed"
    failed_names = {item["name"] for item in report["results"] if item["status"] == "failed"}
    assert "frequency_selectivity_floor" in failed_names


def test_visual_sync_not_applicable_when_camera_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {"strict": True},
        "isaac": {"camera": {"enabled": False}},
    }
    rows = _base_rows(2)
    csi = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [1.1 + 0.1j, 2.1 + 0.1j]], dtype=np.complex64)
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1], config)

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    visual = next(item for item in report["results"] if item["name"] == "visual_sync_check")
    assert visual["status"] == "not_applicable"


def test_visual_sync_passes_with_matching_jsonl_and_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {"strict": True},
        "isaac": {"camera": {"enabled": True, "output_subdir": "renders"}},
    }
    rows = _base_rows(2)
    rows[0]["image_path"] = "renders/frame_0000.png"
    rows[1]["image_path"] = "renders/frame_0001.png"
    csi = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [1.1 + 0.1j, 2.1 + 0.1j]], dtype=np.complex64)
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1], config, renders_dir="renders")
    renders = run_root / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    (renders / "frame_0000.png").write_bytes(b"rgb0")
    (renders / "frame_0001.png").write_bytes(b"rgb1")

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    visual = next(item for item in report["results"] if item["name"] == "visual_sync_check")
    assert visual["status"] == "passed"


def test_visual_sync_fails_on_missing_or_empty_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = {
        "scenario": {"scene_usd": "dummy.usd", "id": "hospital"},
        "qa": {"strict": True},
        "isaac": {"camera": {"enabled": True, "output_subdir": "renders"}},
    }
    rows = _base_rows(2)
    rows[0]["image_path"] = "renders/frame_0000.png"
    rows[1]["image_path"] = "renders/frame_0001.png"
    csi = np.asarray([[1.0 + 0.0j, 2.0 + 0.0j], [1.1 + 0.1j, 2.1 + 0.1j]], dtype=np.complex64)
    run_root = _write_artifacts(tmp_path, rows, csi, [0, 1], config, renders_dir="renders")
    renders = run_root / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    (renders / "frame_0000.png").write_bytes(b"")
    # frame_0001.png intentionally missing

    monkeypatch.setattr(
        "isaacsim_sionna.qa.dataset_validator.extract_mesh_aabbs_from_usd_file",
        lambda scene_usd, max_meshes=None: [
            type("M", (), {"center_xyz": [2.5, 0.0, 1.5], "half_extent_xyz": [10.0, 10.0, 10.0], "prim_path": "/A"})
        ],
    )

    report = validate_run(run_root=run_root, config=config, strict=True, write_manifest=False)
    assert report["status"] == "failed"
    visual = next(item for item in report["results"] if item["name"] == "visual_sync_check")
    assert visual["status"] == "failed"
    assert visual["details"]["count_mismatch"] is True
    assert "renders/frame_0001.png" in visual["details"]["missing_files"]
    assert "renders/frame_0000.png" in visual["details"]["zero_byte_files"]
