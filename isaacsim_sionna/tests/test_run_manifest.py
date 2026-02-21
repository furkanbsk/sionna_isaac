from __future__ import annotations

import json
from pathlib import Path

from isaacsim_sionna.utils.run_manifest import build_manifest, collect_git_info, merge_qa_into_manifest, write_manifest


def test_collect_git_info_graceful_outside_repo(tmp_path: Path) -> None:
    info = collect_git_info(cwd=tmp_path)
    assert "available" in info
    assert "commit" in info
    assert "is_dirty" in info


def test_write_manifest_roundtrip(tmp_path: Path) -> None:
    manifest = build_manifest(
        timestamp_utc="2026-02-20T00:00:00Z",
        project_name="demo",
        seed=42,
        config_hash="abc123",
        hash_algo="sha256",
        git_info={"available": True, "commit": "deadbeef", "is_dirty": False},
        config={"project": {"seed": 42}},
        seeded_libraries=["random", "numpy"],
        outputs={"samples_jsonl": "samples.jsonl", "num_samples": 1, "samples_sha256": "bead"},
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, manifest)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["config_hash"] == "abc123"
    assert loaded["seed"] == 42
    assert loaded["outputs"]["num_samples"] == 1


def test_merge_qa_into_manifest(tmp_path: Path) -> None:
    manifest = build_manifest(
        timestamp_utc="2026-02-20T00:00:00Z",
        project_name="demo",
        seed=42,
        config_hash="abc123",
        hash_algo="sha256",
        git_info={"available": True, "commit": "deadbeef", "is_dirty": False},
        config={"project": {"seed": 42}},
        seeded_libraries=["random", "numpy"],
        outputs={"samples_jsonl": "samples.jsonl", "num_samples": 1, "samples_sha256": "bead"},
    )
    out = tmp_path / "manifest.json"
    write_manifest(out, manifest)
    merge_qa_into_manifest(out, {"status": "passed", "summary": {"num_failures": 0}})
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["qa_status"] == "passed"
    assert loaded["qa"]["summary"]["num_failures"] == 0
