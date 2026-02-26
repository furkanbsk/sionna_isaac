from __future__ import annotations

import numpy as np

from isaacsim_sionna.bridge.sionna_adapter import SionnaAdapter


class _Arr:
    def __init__(self, a):
        self._a = np.asarray(a)

    def numpy(self):
        return self._a


class _FakePaths:
    synthetic_array = True

    def __init__(self):
        # [max_depth, num_tgt, num_src, num_paths, 3]
        vertices = np.zeros((3, 1, 1, 1, 3), dtype=np.float64)
        vertices[0, 0, 0, 0] = [1.0, 0.0, 1.5]
        vertices[1, 0, 0, 0] = [1.5, 1.0, 1.5]
        self.vertices = _Arr(vertices)

        # [num_tgt, num_src, num_paths]
        self.valid = _Arr(np.array([[[True]]], dtype=bool))
        # [max_depth, num_tgt, num_src, num_paths]
        # two bounces: SPECULAR(1), DIFFRACTION(8), then NONE(0)
        self.interactions = _Arr(np.array([[[[1]]], [[[8]]], [[[0]]]], dtype=np.int32))

        # Expected by adapter as .numpy().T -> [num,3]
        self.sources = _Arr(np.array([[0.0], [0.0], [1.5]], dtype=np.float64))
        self.targets = _Arr(np.array([[2.0], [2.0], [1.5]], dtype=np.float64))


def test_extract_path_polylines_contains_bounces_and_interaction_types() -> None:
    adapter = SionnaAdapter(
        {
            "radio": {"carrier_hz": 3.5e9, "bandwidth_hz": 20e6, "num_subcarriers": 64},
            "isaac": {"visualize_rays": True, "ray_viz": {"max_paths_to_draw": 8}},
        }
    )
    out = adapter._extract_path_polylines(_FakePaths())  # pylint: disable=protected-access
    assert len(out) == 1
    p = out[0]
    assert p["is_los"] is False
    assert p["interaction_types"] == [1, 8]
    # TX + 2 bounce vertices + RX
    assert len(p["points_xyz"]) == 4
    assert p["num_segments"] == 3
