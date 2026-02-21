#!/usr/bin/env python3
"""Render CSI time-frequency heatmaps from HDF5 output."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CSI magnitude/phase heatmaps from csi_tensors.h5")
    parser.add_argument(
        "--h5",
        default="isaacsim_sionna/data/raw/csi_tensors.h5",
        help="Path to HDF5 tensor store with /frames/csi_c64",
    )
    parser.add_argument(
        "--output",
        default="isaacsim_sionna/data/raw/csi_heatmap.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=0,
        help="Start row (inclusive) in /frames/csi_c64",
    )
    parser.add_argument(
        "--end-row",
        type=int,
        default=None,
        help="End row (exclusive) in /frames/csi_c64. Default: all rows",
    )
    parser.add_argument(
        "--unwrap-phase",
        action="store_true",
        help="Apply phase unwrapping along time axis",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Render only the raw magnitude panel.",
    )
    parser.add_argument(
        "--no-phase",
        action="store_true",
        help="Do not render phase panel.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="Output image DPI",
    )
    return parser.parse_args()


def load_csi_matrix(h5_path: Path, start_row: int = 0, end_row: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load complex CSI matrix and corresponding frame indices."""
    with h5py.File(str(h5_path), "r") as f:
        if "frames" not in f or "csi_c64" not in f["frames"]:
            raise ValueError("Missing /frames/csi_c64 in HDF5 file")
        csi_ds = f["frames"]["csi_c64"]
        frame_ds = f["frames"].get("frame_idx")
        nrows = int(csi_ds.shape[0])
        s0 = max(0, int(start_row))
        s1 = nrows if end_row is None else min(int(end_row), nrows)
        if s1 <= s0:
            raise ValueError(f"Invalid row window: start={s0}, end={s1}, total={nrows}")
        csi = np.asarray(csi_ds[s0:s1])
        if frame_ds is None:
            frame_idx = np.arange(s0, s1, dtype=np.int32)
        else:
            frame_idx = np.asarray(frame_ds[s0:s1], dtype=np.int32)
    return csi, frame_idx


def build_heatmap_arrays(csi: np.ndarray, unwrap_phase: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return magnitude(dB) and phase arrays with shape [subcarrier, frame]."""
    if csi.ndim != 2:
        raise ValueError(f"Expected CSI rank-2 [frame, subcarrier], got shape={csi.shape}")
    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(csi), 1e-12))
    phase = np.angle(csi)
    if unwrap_phase:
        phase = np.unwrap(phase, axis=0)
    return magnitude_db.T, phase.T


def render_heatmap(
    magnitude_db_raw: np.ndarray,
    magnitude_db_norm: np.ndarray,
    phase: np.ndarray,
    frame_idx: np.ndarray,
    output_path: Path,
    dpi: int = 240,
    include_raw: bool = True,
    include_norm: bool = True,
    include_phase: bool = True,
) -> None:
    """Render and save CSI heatmap figure.

    Default behavior includes raw magnitude, normalized magnitude, and phase.
    """
    if magnitude_db_raw.shape != phase.shape or magnitude_db_norm.shape != phase.shape:
        raise ValueError("Raw magnitude, normalized magnitude, and phase arrays must have identical shape")
    if magnitude_db_raw.shape[1] != int(frame_idx.shape[0]):
        raise ValueError("Frame index length mismatch")
    if not (include_raw or include_norm or include_phase):
        raise ValueError("At least one panel must be enabled")

    x0 = float(frame_idx[0])
    x1 = float(frame_idx[-1]) if frame_idx.size > 1 else float(frame_idx[0] + 1)
    y0 = 0.0
    y1 = float(magnitude_db_raw.shape[0] - 1)
    extent = [x0, x1, y0, y1]

    panels = []
    if include_raw:
        panels.append("raw")
    if include_norm:
        panels.append("norm")
    if include_phase:
        panels.append("phase")

    fig_h = max(4.0, 3.2 * len(panels))
    fig, axes = plt.subplots(len(panels), 1, figsize=(14, fig_h), sharex=True, constrained_layout=True)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    fig.suptitle("CSI Time-Frequency Heatmap")

    ax_i = 0
    if include_raw:
        ax = axes[ax_i]
        im = ax.imshow(
            magnitude_db_raw,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
        ax.set_ylabel("Subcarrier Index")
        ax.set_title("Raw Magnitude [dB]")
        cb = fig.colorbar(im, ax=ax, orientation="vertical")
        cb.set_label("dB")
        ax_i += 1

    if include_norm:
        ax = axes[ax_i]
        im = ax.imshow(
            magnitude_db_norm,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="viridis",
        )
        ax.set_ylabel("Subcarrier Index")
        ax.set_title("Normalized Magnitude [dB]")
        cb = fig.colorbar(im, ax=ax, orientation="vertical")
        cb.set_label("dB")
        ax_i += 1

    if include_phase:
        ax = axes[ax_i]
        im = ax.imshow(
            phase,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="twilight",
        )
        ax.set_ylabel("Subcarrier Index")
        ax.set_title("Phase [rad]")
        cb = fig.colorbar(im, ax=ax, orientation="vertical")
        cb.set_label("rad")

    axes[-1].set_xlabel("Frame Index")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=int(max(72, dpi)))
    plt.close(fig)


def main() -> int:
    args = parse_args()
    h5_path = Path(args.h5)
    if not h5_path.exists():
        print(f"HDF5 file not found: {h5_path}")
        return 1

    csi, frame_idx = load_csi_matrix(h5_path, start_row=args.start_row, end_row=args.end_row)
    magnitude_db_raw, phase = build_heatmap_arrays(csi, unwrap_phase=bool(args.unwrap_phase))
    magnitude_db_norm = magnitude_db_raw - np.mean(magnitude_db_raw, axis=0, keepdims=True)
    include_raw = True
    include_norm = not bool(args.raw_only)
    include_phase = not bool(args.no_phase)
    out = Path(args.output)
    render_heatmap(
        magnitude_db_raw,
        magnitude_db_norm,
        phase,
        frame_idx,
        out,
        dpi=int(args.dpi),
        include_raw=include_raw,
        include_norm=include_norm,
        include_phase=include_phase,
    )
    print(f"Saved heatmap: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
