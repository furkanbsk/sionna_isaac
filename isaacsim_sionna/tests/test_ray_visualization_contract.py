from __future__ import annotations

from isaacsim_sionna.bridge import pipeline as pipeline_mod


class _FakeIsaac:
    def __init__(self, config):
        self.config = config
        self.frame = -1
        self.drawn = []

    def start(self):
        return None

    def stop(self):
        return None

    def step(self):
        self.frame += 1

    def get_state(self):
        return {
            "timestamp_sim": float(self.frame) / 30.0,
            "frame_idx": self.frame,
            "tx_pose": {"pos_xyz": [0.0, 0.0, 1.5]},
            "rx_pose": {"pos_xyz": [1.0, 0.0, 1.5]},
            "actor_poses": [],
        }

    def get_sionna_geometry_proxy(self):
        return {"geometry_mode": "aabb", "mesh_aabbs": [{"prim_path": "/A", "center_xyz": [0, 0, 0], "half_extent_xyz": [1, 1, 1]}]}

    def capture_rgb(self, frame_idx):
        _ = frame_idx
        return None

    def ray_visualization_enabled(self):
        return bool((self.config.get("isaac") or {}).get("visualize_rays", False))

    def render_paths(self, path_geometry):
        self.drawn.append(path_geometry)


class _FakeWriter:
    last = None

    def __init__(self, config):
        self.config = config
        self.rows = []
        _FakeWriter.last = self

    def open(self, run_context=None):
        _ = run_context

    def write(self, frame_idx, state, snapshot):
        self.rows.append((frame_idx, state, snapshot))

    def set_runtime_metrics(self, metrics):
        _ = metrics

    def close(self):
        return None


class _FakeSionna:
    def __init__(self, config):
        self.config = config
        self._geom = [{"points_xyz": [[0.0, 0.0, 1.5], [1.0, 0.0, 1.5]], "is_los": True}]

    def initialize(self, geometry_proxy):
        _ = geometry_proxy

    def update_dynamic_state(self, state):
        _ = state

    def compute_snapshot(self, frame_idx=None):
        return {
            "status": "ok",
            "num_paths": 1,
            "csi_re": [1.0],
            "csi_im": [0.0],
            "a_re": [0.1],
            "a_im": [0.0],
            "tau_s": [1e-9],
            "frame_idx": frame_idx,
        }

    def get_last_path_geometry(self):
        return list(self._geom)

    def get_init_metrics(self):
        return {"geometry_prep_ms": 1.0}


def test_pipeline_forwards_path_geometry_when_enabled(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "IsaacAdapter", _FakeIsaac)
    monkeypatch.setattr(pipeline_mod, "CsiWriter", _FakeWriter)

    import isaacsim_sionna.bridge.sionna_adapter as sionna_mod

    monkeypatch.setattr(sionna_mod, "SionnaAdapter", _FakeSionna)

    cfg = {
        "project": {"seed": 1},
        "isaac": {"visualize_rays": True},
        "runtime": {
            "max_frames": 6,
            "isaac_fps": 60,
            "radio_update_rate_hz": 20,
        },
    }
    p = pipeline_mod.Pipeline(cfg)
    p.run()

    assert len(p.isaac.drawn) == 2
    assert p.isaac.drawn[0][0]["is_los"] is True


def test_pipeline_does_not_forward_path_geometry_when_disabled(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "IsaacAdapter", _FakeIsaac)
    monkeypatch.setattr(pipeline_mod, "CsiWriter", _FakeWriter)

    import isaacsim_sionna.bridge.sionna_adapter as sionna_mod

    monkeypatch.setattr(sionna_mod, "SionnaAdapter", _FakeSionna)

    cfg = {
        "project": {"seed": 1},
        "isaac": {"visualize_rays": False},
        "runtime": {
            "max_frames": 6,
            "isaac_fps": 60,
            "radio_update_rate_hz": 20,
        },
    }
    p = pipeline_mod.Pipeline(cfg)
    p.run()

    assert p.isaac.drawn == []

