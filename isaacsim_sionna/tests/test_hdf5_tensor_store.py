from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from isaacsim_sionna.exporters.hdf5_tensor_store import Hdf5TensorStore


@pytest.mark.skipif(__import__('importlib').util.find_spec('h5py') is None, reason='h5py not installed')
def test_hdf5_tensor_store_roundtrip_complex(tmp_path: Path) -> None:
    store = Hdf5TensorStore(out_root=tmp_path, rel_path='csi_tensors.h5', dtype='complex64', compression=None)
    store.open()

    snap0 = {
        'csi_re': [1.0, 2.0, 3.0],
        'csi_im': [0.5, 1.5, -0.5],
        'num_paths': 2,
        'a_re': [0.1, 0.2],
        'a_im': [0.3, 0.4],
        'tau_s': [1e-9, 2e-9],
    }
    snap1 = {
        'csi_re': [4.0, 5.0, 6.0],
        'csi_im': [0.0, -1.0, 2.0],
        'num_paths': 1,
        'a_re': [0.9],
        'a_im': [0.8],
        'tau_s': [3e-9],
    }

    ref0 = store.append(frame_idx=0, timestamp_sim=0.1, snapshot=snap0)
    ref1 = store.append(frame_idx=1, timestamp_sim=0.2, snapshot=snap1)
    meta = store.close()

    assert ref0 is not None and ref0['row'] == 0
    assert ref1 is not None and ref1['row'] == 1
    assert meta['tensor_store_rows'] == 2
    assert meta['tensor_store_sha256'] is not None

    import h5py

    with h5py.File(str(tmp_path / 'csi_tensors.h5'), 'r') as h5:
        csi = h5['frames']['csi_c64'][:]
        assert csi.shape == (2, 3)
        assert csi.dtype == np.complex64
        np.testing.assert_array_equal(csi[0], np.asarray([1 + 0.5j, 2 + 1.5j, 3 - 0.5j], dtype=np.complex64))
        np.testing.assert_array_equal(csi[1], np.asarray([4 + 0j, 5 - 1j, 6 + 2j], dtype=np.complex64))
        assert int(h5['frames']['num_paths'][0]) == 2
        assert int(h5['frames']['num_paths'][1]) == 1
