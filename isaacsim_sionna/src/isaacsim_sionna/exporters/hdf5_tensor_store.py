"""Append-only HDF5 tensor store for CSI snapshots."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class Hdf5TensorStore:
    """Stores per-frame CSI tensors and path arrays in HDF5."""

    def __init__(
        self,
        out_root: Path,
        rel_path: str = "csi_tensors.h5",
        compression: str | None = "gzip",
        chunk_frames: int = 64,
        dtype: str = "complex64",
    ):
        self.out_root = Path(out_root)
        self.rel_path = str(rel_path)
        self.path = self.out_root / self.rel_path
        self.compression = compression
        self.chunk_frames = int(max(1, chunk_frames))
        self.dtype = np.complex64 if str(dtype).lower() == "complex64" else np.complex128

        self._h5 = None
        self._frames = None
        self._csi = None
        self._timestamp = None
        self._frame_idx = None
        self._num_paths = None
        self._a_re = None
        self._a_im = None
        self._tau = None
        self._num_rows = 0

    def open(self) -> None:
        try:
            import h5py  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pragma: no cover - env dependent
            raise RuntimeError("h5py is required for HDF5 tensor storage") from exc

        self.out_root.mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(str(self.path), "w")
        self._frames = self._h5.create_group("frames")
        self._h5.create_group("meta")
        self._num_rows = 0

    def _ensure_datasets(self, num_subcarriers: int) -> None:
        if self._frames is None:
            raise RuntimeError("Tensor store is not open")
        if self._csi is not None:
            if int(self._csi.shape[1]) != int(num_subcarriers):
                raise ValueError(
                    f"Inconsistent CSI width. Expected {self._csi.shape[1]}, got {num_subcarriers}"
                )
            return

        import h5py  # pylint: disable=import-outside-toplevel

        chunks = (self.chunk_frames, int(num_subcarriers))
        self._csi = self._frames.create_dataset(
            "csi_c64",
            shape=(0, int(num_subcarriers)),
            maxshape=(None, int(num_subcarriers)),
            chunks=chunks,
            compression=self.compression,
            dtype=self.dtype,
        )
        self._timestamp = self._frames.create_dataset(
            "timestamp_sim",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            compression=self.compression,
            dtype=np.float64,
        )
        self._frame_idx = self._frames.create_dataset(
            "frame_idx",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            compression=self.compression,
            dtype=np.int32,
        )
        self._num_paths = self._frames.create_dataset(
            "num_paths",
            shape=(0,),
            maxshape=(None,),
            chunks=(self.chunk_frames,),
            compression=self.compression,
            dtype=np.int32,
        )

        vlen_f32 = h5py.vlen_dtype(np.dtype("float32"))
        self._a_re = self._frames.create_dataset("a_re_f32", shape=(0,), maxshape=(None,), dtype=vlen_f32)
        self._a_im = self._frames.create_dataset("a_im_f32", shape=(0,), maxshape=(None,), dtype=vlen_f32)
        self._tau = self._frames.create_dataset("tau_s_f32", shape=(0,), maxshape=(None,), dtype=vlen_f32)

    def append(self, frame_idx: int, timestamp_sim: float | None, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        csi_re = snapshot.get("csi_re") or []
        csi_im = snapshot.get("csi_im") or []
        if len(csi_re) == 0 or len(csi_im) == 0:
            logger.warning("Skipping HDF5 frame %d: empty CSI data", frame_idx)
            return None

        if len(csi_re) != len(csi_im):
            raise ValueError("csi_re and csi_im length mismatch")

        self._ensure_datasets(len(csi_re))
        row = int(self._num_rows)

        csi = np.asarray(csi_re, dtype=np.float32) + 1j * np.asarray(csi_im, dtype=np.float32)

        for ds in [self._csi, self._timestamp, self._frame_idx, self._num_paths, self._a_re, self._a_im, self._tau]:
            ds.resize((row + 1, *ds.shape[1:]))

        self._csi[row, :] = csi.astype(self.dtype)
        self._timestamp[row] = float(timestamp_sim) if timestamp_sim is not None else np.nan
        self._frame_idx[row] = int(frame_idx)
        self._num_paths[row] = int(snapshot.get("num_paths", 0))
        self._a_re[row] = np.asarray(snapshot.get("a_re") or [], dtype=np.float32)
        self._a_im[row] = np.asarray(snapshot.get("a_im") or [], dtype=np.float32)
        self._tau[row] = np.asarray(snapshot.get("tau_s") or [], dtype=np.float32)

        self._num_rows += 1
        return {
            "file": self.rel_path,
            "group": "/frames",
            "row": row,
            "dataset": "csi_c64",
        }

    def write_metadata(self, meta: dict[str, Any]) -> None:
        if self._h5 is None:
            return
        grp = self._h5["meta"]
        for k, v in meta.items():
            grp.attrs[str(k)] = str(v)

    def close(self) -> dict[str, Any]:
        if self._h5 is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None

        if not self.path.exists():
            return {"tensor_store_sha256": None, "tensor_store_rows": int(self._num_rows)}

        try:
            sha = hashlib.sha256()
            with open(self.path, "rb") as fh:
                for chunk in iter(lambda: fh.read(8192), b""):
                    sha.update(chunk)
            sha_hex = sha.hexdigest()
        except Exception as exc:
            logger.warning("Failed to compute SHA256 for %s: %s", self.path, exc)
            sha_hex = None
        return {
            "tensor_store_sha256": sha_hex,
            "tensor_store_rows": int(self._num_rows),
            "tensor_store_file": self.rel_path,
        }
