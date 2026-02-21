"""CSI writer placeholder.

Swap this for HDF5/Zarr implementation once schema is fixed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

from isaacsim_sionna.exporters.hdf5_tensor_store import Hdf5TensorStore
from isaacsim_sionna.utils.run_manifest import build_manifest, write_manifest


class CsiWriter:
    """Append-only JSONL scaffold writer."""

    def __init__(self, config: dict):
        self.config = config
        self.labels_cfg = config.get("labels", {})
        self.storage_cfg = config.get("storage", {})
        self.tensor_cfg = (self.storage_cfg.get("tensor_store") or {}) if isinstance(self.storage_cfg, dict) else {}
        isaac_cfg = config.get("isaac", {}) if isinstance(config, dict) else {}
        camera_cfg = isaac_cfg.get("camera", {}) if isinstance(isaac_cfg, dict) else {}
        self._camera_enabled = bool(camera_cfg.get("enabled", False))
        self._renders_subdir = str(camera_cfg.get("output_subdir", "renders"))
        self._fp = None
        self._manifest_path: pathlib.Path | None = None
        self._manifest: dict[str, Any] | None = None
        self._num_samples = 0
        self._num_rendered_frames = 0
        self._samples_hash = hashlib.sha256()
        self._tensor_store: Hdf5TensorStore | None = None
        self._runtime_metrics: dict[str, Any] | None = None

    def _resolve_activity_label(self, frame_idx: int) -> str | None:
        schedule = self.labels_cfg.get("activity_schedule")
        if isinstance(schedule, list):
            for item in schedule:
                if not isinstance(item, dict):
                    continue
                start = int(item.get("start_frame", 0))
                end = int(item.get("end_frame", start))
                if start <= int(frame_idx) <= end:
                    label = item.get("label")
                    return None if label is None else str(label)

        label = self.labels_cfg.get("activity_label")
        if label is not None:
            return str(label)
        return None

    def open(self, run_context: dict[str, Any] | None = None) -> None:
        out_root = pathlib.Path(self.config.get("project", {}).get("output_root", "isaacsim_sionna/data/raw"))
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "samples.jsonl"
        self._fp = out_path.open("w", encoding="utf-8")
        self._manifest_path = out_root / "manifest.json"
        self._num_samples = 0
        self._num_rendered_frames = 0
        self._samples_hash = hashlib.sha256()
        self._runtime_metrics = None
        print(f"[CsiWriter] writing {out_path}")

        tensor_enabled = bool(self.tensor_cfg.get("enabled", False))
        if tensor_enabled:
            tensor_path = str(self.tensor_cfg.get("path", "csi_tensors.h5"))
            tensor_dtype = str(self.tensor_cfg.get("dtype", "complex64"))
            tensor_compression = self.tensor_cfg.get("compression", "gzip")
            tensor_chunks = int(self.tensor_cfg.get("chunk_frames", 64))
            self._tensor_store = Hdf5TensorStore(
                out_root=out_root,
                rel_path=tensor_path,
                dtype=tensor_dtype,
                compression=tensor_compression,
                chunk_frames=tensor_chunks,
            )
            self._tensor_store.open()

        if run_context is not None:
            outputs = {
                "samples_jsonl": "samples.jsonl",
                "num_samples": 0,
                "samples_sha256": None,
            }
            if self._camera_enabled:
                outputs["renders_dir"] = self._renders_subdir
                outputs["num_rendered_frames"] = 0
            if self._tensor_store is not None:
                outputs["tensor_store_file"] = self._tensor_store.rel_path
                outputs["tensor_store_sha256"] = None
                outputs["tensor_store_rows"] = 0
            self._manifest = build_manifest(
                timestamp_utc=run_context["timestamp_utc"],
                project_name=run_context["project_name"],
                seed=run_context["seed"],
                config_hash=run_context["config_hash"],
                hash_algo=run_context["hash_algo"],
                git_info=run_context["git"],
                config=run_context["config"],
                seeded_libraries=run_context["runtime"].get("seeded_libraries", []),
                outputs=outputs,
            )
            if self._tensor_store is not None:
                self._tensor_store.write_metadata(
                    {
                        "config_hash": run_context["config_hash"],
                        "seed": run_context["seed"],
                        "project_name": run_context["project_name"],
                    }
                )
            write_manifest(self._manifest_path, self._manifest)

    def set_runtime_metrics(self, metrics: dict[str, Any]) -> None:
        self._runtime_metrics = dict(metrics)

    def write(self, frame_idx: int, state: dict, snapshot: dict, render_ref: dict[str, Any] | None = None) -> None:
        tensor_ref = None
        snapshot_for_row = dict(snapshot)
        if self._tensor_store is not None:
            tensor_ref = self._tensor_store.append(
                frame_idx=int(frame_idx),
                timestamp_sim=state.get("timestamp_sim"),
                snapshot=snapshot,
            )
            # Keep JSONL as an index ledger; bulky tensors live in HDF5.
            for k in ["csi_re", "csi_im", "a_re", "a_im", "tau_s"]:
                snapshot_for_row.pop(k, None)

        row = {
            "frame_idx": frame_idx,
            "timestamp_sim": state.get("timestamp_sim"),
            "actor_poses": state.get("actor_poses", []),
            "state": state,
            "snapshot": snapshot_for_row,
            "image_path": None,
        }
        if tensor_ref is not None:
            row["tensor_ref"] = tensor_ref
        if render_ref is not None:
            row["render_ref"] = dict(render_ref)
            image_path = render_ref.get("file")
            if image_path is not None:
                row["image_path"] = str(image_path)
                self._num_rendered_frames += 1
        activity_label = self._resolve_activity_label(frame_idx)
        if activity_label is not None:
            row["activity_label"] = activity_label
        assert self._fp is not None
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        self._fp.write(payload)
        self._samples_hash.update(payload.encode("utf-8"))
        self._num_samples += 1

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        tensor_meta = {}
        if self._tensor_store is not None:
            tensor_meta = self._tensor_store.close()
            self._tensor_store = None
        if self._manifest is not None and self._manifest_path is not None:
            self._manifest["outputs"]["num_samples"] = int(self._num_samples)
            if self._camera_enabled:
                self._manifest["outputs"]["num_rendered_frames"] = int(self._num_rendered_frames)
            self._manifest["outputs"]["samples_sha256"] = self._samples_hash.hexdigest()
            for key, value in tensor_meta.items():
                self._manifest["outputs"][key] = value
            if self._runtime_metrics is not None:
                self._manifest.setdefault("runtime", {})["performance"] = self._runtime_metrics
            write_manifest(self._manifest_path, self._manifest)
