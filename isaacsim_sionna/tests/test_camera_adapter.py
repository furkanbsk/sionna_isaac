from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

from isaacsim_sionna.bridge.camera_adapter import CameraAdapter


def _install_fake_replicator(monkeypatch):
    class _Annotator:
        def __init__(self):
            self._attached = False

        def attach(self, items):
            _ = items
            self._attached = True

        def detach(self, items):
            _ = items
            self._attached = False

        def get_data(self):
            if not self._attached:
                return None
            return np.zeros((8, 16, 4), dtype=np.uint8)

    class _AnnotatorRegistry:
        @staticmethod
        def get_annotator(name):
            _ = name
            return _Annotator()

    class _Create:
        @staticmethod
        def camera(**kwargs):
            return {"camera": kwargs}

        @staticmethod
        def render_product(camera, resolution):
            return {"camera": camera, "resolution": resolution}

    class _Orchestrator:
        @staticmethod
        def step():
            return None

    rep = types.SimpleNamespace(
        create=_Create(),
        AnnotatorRegistry=_AnnotatorRegistry,
        orchestrator=_Orchestrator(),
    )

    omni = types.ModuleType("omni")
    replicator = types.ModuleType("omni.replicator")
    core = types.ModuleType("omni.replicator.core")
    core.create = rep.create
    core.AnnotatorRegistry = rep.AnnotatorRegistry
    core.orchestrator = rep.orchestrator

    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)


def test_camera_adapter_parses_schema_defaults(tmp_path: Path) -> None:
    cfg = {
        "project": {"output_root": str(tmp_path)},
        "isaac": {
            "camera": {
                "enabled": True,
                "backend": "not_supported",
                "name": "cam0",
                "prim_path": "/World/cam0",
                "position_xyz": [1, 2, 3],
                "target_xyz": [4, 5, 6],
                "resolution": [320, 240],
                "format": "png",
                "warmup_steps": 2,
            }
        },
    }
    adapter = CameraAdapter(cfg)
    assert adapter.enabled is True
    assert adapter.backend == "replicator"
    assert adapter.camera_name == "cam0"
    assert adapter.prim_path == "/World/cam0"
    assert adapter.width == 320
    assert adapter.height == 240
    assert adapter.warmup_steps == 2


def test_camera_adapter_initialize_and_capture(monkeypatch, tmp_path: Path) -> None:
    _install_fake_replicator(monkeypatch)
    cfg = {
        "project": {"output_root": str(tmp_path)},
        "isaac": {
            "camera": {
                "enabled": True,
                "name": "cam1",
                "prim_path": "/World/cam1",
                "resolution": [16, 8],
                "output_subdir": "renders",
            }
        },
    }
    adapter = CameraAdapter(cfg)

    def _save(path, rgb, image_format):
        _ = rgb, image_format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr(CameraAdapter, "_save_image", staticmethod(_save))

    assert adapter.initialize() is True
    ref = adapter.capture(frame_idx=3)
    assert ref is not None
    assert ref["camera_name"] == "cam1"
    assert ref["camera_prim_path"] == "/World/cam1"
    out = tmp_path / ref["file"]
    assert out.exists()
    adapter.close()


def test_camera_adapter_capture_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    class _Annotator:
        def __init__(self):
            self._attached = False
            self.calls = 0

        def attach(self, items):
            _ = items
            self._attached = True

        def detach(self, items):
            _ = items
            self._attached = False

        def get_data(self):
            if not self._attached:
                return None
            self.calls += 1
            if self.calls < 2:
                return None
            return np.zeros((8, 16, 4), dtype=np.uint8)

    class _Create:
        @staticmethod
        def camera(**kwargs):
            return {"camera": kwargs}

        @staticmethod
        def render_product(camera, resolution):
            return {"camera": camera, "resolution": resolution}

    class _Orchestrator:
        calls = 0

        @classmethod
        def step(cls):
            cls.calls += 1
            return None

    annotator = _Annotator()

    class _AnnotatorRegistry:
        @staticmethod
        def get_annotator(name):
            _ = name
            return annotator

    omni = types.ModuleType("omni")
    replicator = types.ModuleType("omni.replicator")
    core = types.ModuleType("omni.replicator.core")
    core.create = _Create()
    core.AnnotatorRegistry = _AnnotatorRegistry
    core.orchestrator = _Orchestrator

    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.replicator", replicator)
    monkeypatch.setitem(sys.modules, "omni.replicator.core", core)

    cfg = {
        "project": {"output_root": str(tmp_path)},
        "isaac": {
            "camera": {
                "enabled": True,
                "resolution": [16, 8],
                "output_subdir": "renders",
                "capture_max_attempts": 3,
                "capture_timeout_s": 1.0,
            }
        },
    }
    adapter = CameraAdapter(cfg)

    def _save(path, rgb, image_format):
        _ = rgb, image_format
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr(CameraAdapter, "_save_image", staticmethod(_save))

    assert adapter.initialize() is True
    ref = adapter.capture(frame_idx=0)
    assert ref is not None
    assert _Orchestrator.calls >= 2
