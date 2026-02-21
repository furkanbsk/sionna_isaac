from __future__ import annotations

from isaacsim_sionna.bridge import pipeline as pipeline_mod


class _FakeIsaac:
    def __init__(self, config):
        self.config = config
        self.frame = -1
        self.stopped = False

    def start(self):
        return None

    def stop(self):
        self.stopped = True

    def step(self):
        self.frame += 1

    def get_state(self):
        return {
            "timestamp_sim": float(self.frame) / 60.0,
            "frame_idx": self.frame,
            "tx_pose": {"pos_xyz": [0.0, 0.0, 1.5]},
            "rx_pose": {"pos_xyz": [1.0, 0.0, 1.5]},
            "actor_poses": [],
        }

    def get_sionna_geometry_proxy(self):
        return {"geometry_mode": "aabb", "mesh_aabbs": [{"prim_path": "/A", "center_xyz": [0, 0, 0], "half_extent_xyz": [1, 1, 1]}]}

    def capture_rgb(self, frame_idx):
        return {"file": f"renders/frame_{int(frame_idx):04d}.png"}


class _FakeWriter:
    last = None

    def __init__(self, config):
        self.config = config
        self.rows = []
        self.opened = False
        self.closed = False
        self.runtime_metrics = None
        _FakeWriter.last = self

    def open(self, run_context=None):
        _ = run_context
        self.opened = True

    def write(self, frame_idx, state, snapshot):
        self.rows.append((frame_idx, state, snapshot))

    def set_runtime_metrics(self, metrics):
        self.runtime_metrics = metrics

    def close(self):
        self.closed = True


class _FakeSionna:
    def __init__(self, config):
        self.config = config
        self.updates = []

    def initialize(self, geometry_proxy):
        _ = geometry_proxy

    def update_dynamic_state(self, state):
        self.updates.append(state["frame_idx"])

    def compute_snapshot(self, frame_idx=None):
        return {
            "status": "ok",
            "num_paths": 2,
            "csi_re": [1.0, 2.0],
            "csi_im": [0.0, 0.0],
            "a_re": [0.1],
            "a_im": [0.0],
            "tau_s": [1e-9],
            "frame_idx": frame_idx,
        }

    def get_init_metrics(self):
        return {"geometry_prep_ms": 12.5}


def test_pipeline_decimation_by_radio_rate(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "IsaacAdapter", _FakeIsaac)
    monkeypatch.setattr(pipeline_mod, "CsiWriter", _FakeWriter)

    import isaacsim_sionna.bridge.sionna_adapter as sionna_mod

    monkeypatch.setattr(sionna_mod, "SionnaAdapter", _FakeSionna)

    cfg = {
        "project": {"seed": 1},
        "runtime": {
            "max_frames": 12,
            "isaac_fps": 60,
            "radio_update_rate_hz": 10,
        },
    }
    pipeline_mod.Pipeline(cfg).run()

    writer = _FakeWriter.last
    assert writer is not None
    assert writer.opened and writer.closed
    # frame indices 0, 6
    assert [row[0] for row in writer.rows] == [0, 6]


def test_pipeline_decimation_frame_skip_override(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "IsaacAdapter", _FakeIsaac)
    monkeypatch.setattr(pipeline_mod, "CsiWriter", _FakeWriter)

    import isaacsim_sionna.bridge.sionna_adapter as sionna_mod

    monkeypatch.setattr(sionna_mod, "SionnaAdapter", _FakeSionna)

    cfg = {
        "project": {"seed": 1},
        "runtime": {
            "max_frames": 10,
            "isaac_fps": 60,
            "radio_update_rate_hz": 10,
            "frame_skip_ratio": 3,
        },
    }
    pipeline_mod.Pipeline(cfg).run()

    writer = _FakeWriter.last
    assert writer is not None
    # frame indices 0,3,6,9
    assert [row[0] for row in writer.rows] == [0, 3, 6, 9]


def test_pipeline_emits_runtime_performance_metrics(monkeypatch):
    monkeypatch.setattr(pipeline_mod, "IsaacAdapter", _FakeIsaac)
    monkeypatch.setattr(pipeline_mod, "CsiWriter", _FakeWriter)

    import isaacsim_sionna.bridge.sionna_adapter as sionna_mod

    monkeypatch.setattr(sionna_mod, "SionnaAdapter", _FakeSionna)

    cfg = {
        "project": {"seed": 1},
        "runtime": {
            "max_frames": 6,
            "isaac_fps": 60,
            "radio_update_rate_hz": 20,
        },
    }
    pipeline_mod.Pipeline(cfg).run()

    writer = _FakeWriter.last
    assert writer is not None
    perf = writer.runtime_metrics
    assert perf is not None
    assert perf["isaac_step"]["count"] == 6
    assert perf["path_solver"]["count"] == len(writer.rows)
    assert perf["rgb_render"]["count"] == len(writer.rows)
    assert perf["hdf5_write"]["count"] == len(writer.rows)
    assert perf["geometry_prep"]["mean_ms"] == 12.5
    assert perf["num_radio_updates"] == len(writer.rows)
    assert perf["effective_radio_hz"] >= 0.0
