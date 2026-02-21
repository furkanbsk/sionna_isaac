"""Post-run dataset QA validator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_sionna.bridge.usd_to_sionna import compute_global_bbox, extract_mesh_aabbs_from_usd_file
from isaacsim_sionna.utils.run_manifest import merge_qa_into_manifest


@dataclass
class CheckResult:
    name: str
    status: str
    details: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _is_indoor_scene(config: dict[str, Any]) -> bool:
    sid = str((config.get("scenario") or {}).get("id", "")).lower()
    return any(token in sid for token in ["indoor", "hospital", "room", "office", "home"])


def _activity_is_moving(rows: list[dict[str, Any]]) -> bool:
    moving_labels = {"walking", "walk", "running", "run", "falling", "fall", "wave", "sit", "stand"}
    idle_labels = {"idle", "unknown", "none", ""}
    for row in rows:
        label = str(row.get("activity_label", "")).strip().lower()
        if not label:
            continue
        if label in moving_labels:
            return True
        if label not in idle_labels:
            return True
    return False


def _check_path_count(rows: list[dict[str, Any]], config: dict[str, Any]) -> CheckResult:
    num_paths = [int((r.get("snapshot") or {}).get("num_paths", 0)) for r in rows]
    zero_idx = [i for i, n in enumerate(num_paths) if n == 0]
    low_idx = [i for i, n in enumerate(num_paths) if n <= 1]
    collapse_ratio = float(len(low_idx)) / float(max(len(num_paths), 1))
    max_los_ratio = float(((config.get("qa") or {}).get("path_count") or {}).get("max_los_only_ratio", 0.8))

    failed = False
    reasons: list[str] = []
    if zero_idx:
        failed = True
        reasons.append("zero_paths_detected")
    if _is_indoor_scene(config) and collapse_ratio > max_los_ratio:
        failed = True
        reasons.append("los_only_collapse")

    return CheckResult(
        name="path_count_sanity",
        status="failed" if failed else "passed",
        details={
            "num_rows": len(rows),
            "zero_path_frames": zero_idx,
            "los_or_single_path_frames": low_idx,
            "los_or_single_ratio": collapse_ratio,
            "max_los_only_ratio": max_los_ratio,
            "reasons": reasons,
        },
    )


def _resolve_tensor_path(run_root: Path, manifest: dict[str, Any]) -> Path | None:
    out = manifest.get("outputs") or {}
    rel = out.get("tensor_store_file")
    if rel:
        return run_root / str(rel)
    candidate = run_root / "csi_tensors.h5"
    return candidate if candidate.exists() else None


def _check_csi_variance(rows: list[dict[str, Any]], config: dict[str, Any], tensor_path: Path | None) -> CheckResult:
    if tensor_path is None or not tensor_path.exists():
        return CheckResult(name="csi_variance_floor", status="not_applicable", details={"reason": "no_tensor_store"})

    moving = _activity_is_moving(rows)
    if not moving:
        return CheckResult(name="csi_variance_floor", status="not_applicable", details={"reason": "non_moving_activity"})

    import h5py  # pylint: disable=import-outside-toplevel

    with h5py.File(str(tensor_path), "r") as h5:
        if "frames" not in h5 or "csi_c64" not in h5["frames"]:
            return CheckResult(name="csi_variance_floor", status="failed", details={"reason": "missing_csi_c64"})
        csi = np.asarray(h5["frames"]["csi_c64"])

    if csi.shape[0] < 2:
        return CheckResult(name="csi_variance_floor", status="failed", details={"reason": "insufficient_frames"})

    per_frame_delta = np.mean(np.abs(csi[1:] - csi[:-1]), axis=1)
    scale = float(np.median(np.mean(np.abs(csi), axis=1)))
    qa_cfg = config.get("qa") or {}
    var_cfg = qa_cfg.get("csi_variance") or {}
    adaptive_scale = float(var_cfg.get("adaptive_scale", 1e-4))
    abs_floor = float(var_cfg.get("abs_floor", 1e-8))
    floor = max(abs_floor, adaptive_scale * scale)
    p50 = float(np.percentile(per_frame_delta, 50))
    p90 = float(np.percentile(per_frame_delta, 90))
    metric = str(var_cfg.get("metric", "median")).strip().lower()
    if metric == "p90":
        score = p90
    else:
        metric = "median"
        score = p50
    failed = score < floor

    return CheckResult(
        name="csi_variance_floor",
        status="failed" if failed else "passed",
        details={
            "num_frames": int(csi.shape[0]),
            "median_delta": p50,
            "p90_delta": p90,
            "metric": metric,
            "metric_value": score,
            "signal_scale_median": scale,
            "adaptive_floor": floor,
            "adaptive_scale": adaptive_scale,
            "abs_floor": abs_floor,
        },
    )


def _check_frequency_selectivity(config: dict[str, Any], tensor_path: Path | None) -> CheckResult:
    if tensor_path is None or not tensor_path.exists():
        return CheckResult(name="frequency_selectivity_floor", status="not_applicable", details={"reason": "no_tensor_store"})

    import h5py  # pylint: disable=import-outside-toplevel

    with h5py.File(str(tensor_path), "r") as h5:
        if "frames" not in h5 or "csi_c64" not in h5["frames"]:
            return CheckResult(name="frequency_selectivity_floor", status="failed", details={"reason": "missing_csi_c64"})
        csi = np.asarray(h5["frames"]["csi_c64"])

    if csi.shape[1] < 2:
        return CheckResult(name="frequency_selectivity_floor", status="not_applicable", details={"reason": "insufficient_subcarriers"})

    mag = np.abs(csi)
    per_frame_sc_std = np.std(mag, axis=1)
    qa_cfg = config.get("qa") or {}
    fs_cfg = qa_cfg.get("frequency_selectivity") or {}
    abs_floor = float(fs_cfg.get("abs_floor", 1e-6))
    metric = str(fs_cfg.get("metric", "p90")).strip().lower()
    if metric == "median":
        score = float(np.percentile(per_frame_sc_std, 50))
        metric_name = "median"
    else:
        score = float(np.percentile(per_frame_sc_std, 90))
        metric_name = "p90"
    failed = score < abs_floor
    return CheckResult(
        name="frequency_selectivity_floor",
        status="failed" if failed else "passed",
        details={
            "num_frames": int(csi.shape[0]),
            "num_subcarriers": int(csi.shape[1]),
            "metric": metric_name,
            "metric_value": score,
            "abs_floor": abs_floor,
            "median_sc_std": float(np.percentile(per_frame_sc_std, 50)),
            "p90_sc_std": float(np.percentile(per_frame_sc_std, 90)),
        },
    )


def _check_bounds(rows: list[dict[str, Any]], config: dict[str, Any]) -> CheckResult:
    qa_cfg = config.get("qa") or {}
    bounds_cfg = qa_cfg.get("bounds") or {}
    margin = float(bounds_cfg.get("margin_m", 1.0))
    scene_usd = str((config.get("scenario") or {}).get("scene_usd", ""))
    if not scene_usd:
        return CheckResult(name="bounding_sanity", status="not_applicable", details={"reason": "scene_usd_missing"})

    try:
        max_meshes = int(((config.get("isaac") or {}).get("anchors") or {}).get("max_meshes", 256))
        aabbs = extract_mesh_aabbs_from_usd_file(scene_usd, max_meshes=max_meshes)
        scene_bbox = compute_global_bbox(aabbs)
    except ModuleNotFoundError as exc:  # pragma: no cover - env dependent
        if "pxr" in str(exc):
            return CheckResult(name="bounding_sanity", status="not_applicable", details={"reason": "pxr_unavailable"})
        return CheckResult(name="bounding_sanity", status="failed", details={"reason": f"bbox_extract_failed:{exc}"})
    except Exception as exc:  # pragma: no cover - runtime/env dependent
        return CheckResult(name="bounding_sanity", status="failed", details={"reason": f"bbox_extract_failed:{exc}"})

    mn = [float(v) - margin for v in scene_bbox["min_xyz"]]
    mx = [float(v) + margin for v in scene_bbox["max_xyz"]]

    bad_tx = []
    bad_rx = []
    for i, row in enumerate(rows):
        tx = (((row.get("state") or {}).get("tx_pose") or {}).get("pos_xyz"))
        rx = (((row.get("state") or {}).get("rx_pose") or {}).get("pos_xyz"))
        if tx and len(tx) == 3:
            if any(float(tx[j]) < mn[j] or float(tx[j]) > mx[j] for j in range(3)):
                bad_tx.append(i)
        if rx and len(rx) == 3:
            if any(float(rx[j]) < mn[j] or float(rx[j]) > mx[j] for j in range(3)):
                bad_rx.append(i)

    failed = bool(bad_tx or bad_rx)
    return CheckResult(
        name="bounding_sanity",
        status="failed" if failed else "passed",
        details={
            "bbox_min_xyz": mn,
            "bbox_max_xyz": mx,
            "tx_out_of_bounds_frames": bad_tx,
            "rx_out_of_bounds_frames": bad_rx,
        },
    )


def _check_sync(rows: list[dict[str, Any]], manifest: dict[str, Any], run_root: Path) -> CheckResult:
    tensor_path = _resolve_tensor_path(run_root, manifest)
    if tensor_path is None or not tensor_path.exists():
        return CheckResult(name="sync_check", status="not_applicable", details={"reason": "no_tensor_store"})

    import h5py  # pylint: disable=import-outside-toplevel

    missing_ref = []
    bad_index = []
    bad_frame_idx = []

    with h5py.File(str(tensor_path), "r") as h5:
        frames = h5.get("frames")
        if frames is None or "csi_c64" not in frames or "frame_idx" not in frames:
            return CheckResult(name="sync_check", status="failed", details={"reason": "missing_h5_datasets"})
        nrows = int(frames["csi_c64"].shape[0])
        frame_idx_ds = frames["frame_idx"]

        for i, row in enumerate(rows):
            ref = row.get("tensor_ref")
            if ref is None:
                missing_ref.append(i)
                continue
            h5_row = int(ref.get("row", -1))
            if h5_row < 0 or h5_row >= nrows:
                bad_index.append(i)
                continue
            frame_idx = int(row.get("frame_idx", -1))
            if int(frame_idx_ds[h5_row]) != frame_idx:
                bad_frame_idx.append(i)

    failed = bool(missing_ref or bad_index or bad_frame_idx)
    return CheckResult(
        name="sync_check",
        status="failed" if failed else "passed",
        details={
            "missing_tensor_ref_rows": missing_ref,
            "invalid_tensor_row_rows": bad_index,
            "frame_idx_mismatch_rows": bad_frame_idx,
        },
    )


def _make_summary(results: list[CheckResult]) -> dict[str, Any]:
    failures = [r for r in results if r.status == "failed"]
    return {
        "num_checks": len(results),
        "num_failures": len(failures),
        "failed_checks": [r.name for r in failures],
    }


def validate_run(
    run_root: str | Path,
    config: dict[str, Any] | None = None,
    strict: bool = True,
    write_manifest: bool = True,
) -> dict[str, Any]:
    run_root = Path(run_root)
    manifest_path = run_root / "manifest.json"
    samples_path = run_root / "samples.jsonl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not samples_path.exists():
        raise FileNotFoundError(f"Missing samples: {samples_path}")

    manifest = _load_json(manifest_path)
    rows = _load_jsonl(samples_path)
    cfg = config or manifest.get("config") or {}

    tensor_path = _resolve_tensor_path(run_root, manifest)

    checks = [
        _check_path_count(rows, cfg),
        _check_csi_variance(rows, cfg, tensor_path),
        _check_frequency_selectivity(cfg, tensor_path),
        _check_bounds(rows, cfg),
        _check_sync(rows, manifest, run_root),
    ]

    summary = _make_summary(checks)
    failed = summary["num_failures"] > 0
    status = "failed" if failed else "passed"
    qa_report = {
        "status": status,
        "strict_mode": bool(strict),
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "validator_version": "1.0",
        "summary": summary,
        "results": [
            {
                "name": c.name,
                "status": c.status,
                "details": c.details,
            }
            for c in checks
        ],
    }

    if write_manifest:
        merge_qa_into_manifest(manifest_path, qa_report)

    qa_report["exit_code"] = 1 if (failed and strict) else 0
    return qa_report
