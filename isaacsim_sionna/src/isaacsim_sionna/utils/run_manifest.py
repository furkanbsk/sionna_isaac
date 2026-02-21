"""Run manifest helpers."""

from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys
from typing import Any


def _git_cmd(args: list[str], cwd: pathlib.Path) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(["git", *args], cwd=str(cwd), stderr=subprocess.DEVNULL, text=True)
        return True, out.strip()
    except Exception:
        return False, ""


def collect_git_info(cwd: pathlib.Path | None = None) -> dict[str, Any]:
    """Collect git commit metadata, graceful when git is unavailable."""
    root = cwd or pathlib.Path.cwd()
    ok_commit, commit = _git_cmd(["rev-parse", "HEAD"], root)
    ok_dirty, dirty_out = _git_cmd(["status", "--porcelain"], root)
    available = ok_commit or ok_dirty
    return {
        "available": bool(available),
        "commit": commit if ok_commit else None,
        "is_dirty": bool(dirty_out) if ok_dirty else None,
    }


def build_manifest(
    *,
    timestamp_utc: str,
    project_name: str,
    seed: int,
    config_hash: str,
    hash_algo: str,
    git_info: dict[str, Any],
    config: dict[str, Any],
    seeded_libraries: list[str],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON-serializable run manifest dict."""
    return {
        "run_timestamp_utc": timestamp_utc,
        "project_name": project_name,
        "seed": int(seed),
        "config_hash": config_hash,
        "hash_algo": hash_algo,
        "git": git_info,
        "runtime": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "seeded_libraries": list(seeded_libraries),
        },
        "outputs": outputs,
        "config": config,
    }


def write_manifest(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_qa_into_manifest(path: pathlib.Path, qa_report: dict[str, Any]) -> None:
    manifest = read_manifest(path)
    manifest["qa_status"] = str(qa_report.get("status", "failed"))
    manifest["qa"] = qa_report
    write_manifest(path, manifest)
