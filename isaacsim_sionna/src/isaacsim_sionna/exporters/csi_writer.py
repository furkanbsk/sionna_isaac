"""CSI writer placeholder.

Swap this for HDF5/Zarr implementation once schema is fixed.
"""

from __future__ import annotations

import json
import pathlib


class CsiWriter:
    """Append-only JSONL scaffold writer."""

    def __init__(self, config: dict):
        self.config = config
        self._fp = None

    def open(self) -> None:
        out_root = pathlib.Path(self.config.get("project", {}).get("output_root", "isaacsim_sionna/data/raw"))
        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / "samples.jsonl"
        self._fp = out_path.open("w", encoding="utf-8")
        print(f"[CsiWriter] writing {out_path}")

    def write(self, frame_idx: int, state: dict, snapshot: dict) -> None:
        row = {
            "frame_idx": frame_idx,
            "state": state,
            "snapshot": snapshot,
        }
        assert self._fp is not None
        self._fp.write(json.dumps(row) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
