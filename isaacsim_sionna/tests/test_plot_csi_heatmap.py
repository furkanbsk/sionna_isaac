from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import h5py
import numpy as np


def _load_plot_module():
    script = Path("isaacsim_sionna/scripts/plot_csi_heatmap.py").resolve()
    spec = importlib.util.spec_from_file_location("plot_csi_heatmap_mod", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_plot_csi_heatmap_generates_png(tmp_path: Path, monkeypatch) -> None:
    mod = _load_plot_module()
    h5_path = tmp_path / "csi_tensors.h5"
    out_path = tmp_path / "heatmap.png"

    frames = 12
    subcarriers = 64
    rng = np.random.default_rng(123)
    csi = (rng.standard_normal((frames, subcarriers)) + 1j * rng.standard_normal((frames, subcarriers))).astype(np.complex64)

    with h5py.File(str(h5_path), "w") as f:
        g = f.create_group("frames")
        g.create_dataset("csi_c64", data=csi, dtype=np.complex64)
        g.create_dataset("frame_idx", data=np.arange(frames, dtype=np.int32))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_csi_heatmap.py",
            "--h5",
            str(h5_path),
            "--output",
            str(out_path),
            "--unwrap-phase",
        ],
    )

    rc = mod.main()
    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_csi_heatmap_raw_only_mode(tmp_path: Path, monkeypatch) -> None:
    mod = _load_plot_module()
    h5_path = tmp_path / "csi_tensors.h5"
    out_path = tmp_path / "heatmap_raw_only.png"

    frames = 8
    subcarriers = 32
    rng = np.random.default_rng(99)
    csi = (rng.standard_normal((frames, subcarriers)) + 1j * rng.standard_normal((frames, subcarriers))).astype(np.complex64)

    with h5py.File(str(h5_path), "w") as f:
        g = f.create_group("frames")
        g.create_dataset("csi_c64", data=csi, dtype=np.complex64)
        g.create_dataset("frame_idx", data=np.arange(frames, dtype=np.int32))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_csi_heatmap.py",
            "--h5",
            str(h5_path),
            "--output",
            str(out_path),
            "--raw-only",
            "--no-phase",
        ],
    )

    rc = mod.main()
    assert rc == 0
    assert out_path.exists()
    assert out_path.stat().st_size > 0
