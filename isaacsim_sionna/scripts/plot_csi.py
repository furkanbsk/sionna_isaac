#!/usr/bin/env python3
"""Plot CSI magnitude and phase from JSONL pipeline output."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CSI magnitude/phase from samples.jsonl")
    parser.add_argument(
        "--input",
        default="isaacsim_sionna/data/raw/samples.jsonl",
        help="Path to JSONL produced by run_pipeline.py",
    )
    parser.add_argument(
        "--frame-idx",
        type=int,
        default=None,
        help="Frame index to plot. Default: last valid sample.",
    )
    parser.add_argument(
        "--output",
        default="isaacsim_sionna/data/raw/csi_plot.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (optional).",
    )
    return parser.parse_args()


def _load_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _select_row(rows: list[dict[str, Any]], frame_idx: int | None) -> dict[str, Any]:
    valid_rows = []
    for row in rows:
        snapshot = row.get("snapshot", {})
        if snapshot.get("status") != "ok":
            continue
        csi_re = snapshot.get("csi_re", [])
        csi_im = snapshot.get("csi_im", [])
        if not csi_re or not csi_im:
            continue
        valid_rows.append(row)

    if not valid_rows:
        raise ValueError("No valid rows with snapshot.status=ok and non-empty CSI were found.")

    if frame_idx is None:
        return valid_rows[-1]

    for row in valid_rows:
        if int(row.get("frame_idx", -1)) == frame_idx:
            return row
    raise ValueError(f"Requested frame_idx={frame_idx} not found in valid rows.")


def main() -> int:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    rows = _load_rows(input_path)
    row = _select_row(rows, args.frame_idx)

    frame_idx = int(row.get("frame_idx", -1))
    snapshot = row["snapshot"]
    csi_re = np.asarray(snapshot["csi_re"], dtype=np.float64)
    csi_im = np.asarray(snapshot["csi_im"], dtype=np.float64)
    csi = csi_re + 1j * csi_im

    magnitude_db = 20.0 * np.log10(np.maximum(np.abs(csi), 1e-15))
    phase_rad = np.angle(csi)
    subcarriers = np.arange(csi.shape[0], dtype=np.int32)

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(f"CSI Snapshot (frame_idx={frame_idx})")

    ax_mag.plot(subcarriers, magnitude_db, linewidth=1.5)
    ax_mag.set_ylabel("Magnitude [dB]")
    ax_mag.grid(True, alpha=0.3)

    ax_phase.plot(subcarriers, phase_rad, linewidth=1.2)
    ax_phase.set_ylabel("Phase [rad]")
    ax_phase.set_xlabel("Subcarrier Index")
    ax_phase.set_ylim(-math.pi, math.pi)
    ax_phase.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    print(f"Saved plot: {output_path}")

    if args.show:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
