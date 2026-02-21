from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import yaml


def _load_run_pipeline_module():
    script = Path("isaacsim_sionna/scripts/run_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("run_pipeline_mod", script)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_pipeline_validate_after_run_invokes_validator(monkeypatch, tmp_path: Path) -> None:
    mod = _load_run_pipeline_module()

    cfg_path = tmp_path / "cfg.yaml"
    out_root = tmp_path / "out"
    cfg = {"project": {"output_root": str(out_root)}, "runtime": {"max_frames": 1}, "qa": {"strict": True}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class _Pipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "manifest.json").write_text("{}\n", encoding="utf-8")
            (out_root / "samples.jsonl").write_text("", encoding="utf-8")

    calls = {"n": 0}

    def _validate_run(run_root, config, strict, write_manifest):
        calls["n"] += 1
        assert Path(run_root) == out_root
        assert strict is True
        assert write_manifest is True
        return {"status": "passed", "summary": {"num_failures": 0}, "exit_code": 0}

    monkeypatch.setattr(mod, "Pipeline", _Pipeline)
    monkeypatch.setattr("isaacsim_sionna.qa.dataset_validator.validate_run", _validate_run)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(cfg_path), "--validate-after-run"])

    rc = mod.main()
    assert rc == 0
    assert calls["n"] == 1


def test_run_pipeline_validate_after_run_propagates_failure(monkeypatch, tmp_path: Path) -> None:
    mod = _load_run_pipeline_module()

    cfg_path = tmp_path / "cfg.yaml"
    out_root = tmp_path / "out"
    cfg = {"project": {"output_root": str(out_root)}, "runtime": {"max_frames": 1}, "qa": {"strict": True}}
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    class _Pipeline:
        def __init__(self, config):
            self.config = config

        def run(self):
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "manifest.json").write_text("{}\n", encoding="utf-8")
            (out_root / "samples.jsonl").write_text("", encoding="utf-8")

    def _validate_run(run_root, config, strict, write_manifest):
        _ = run_root, config, strict, write_manifest
        return {"status": "failed", "summary": {"num_failures": 2}, "exit_code": 1}

    monkeypatch.setattr(mod, "Pipeline", _Pipeline)
    monkeypatch.setattr("isaacsim_sionna.qa.dataset_validator.validate_run", _validate_run)
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--config", str(cfg_path), "--validate-after-run"])

    rc = mod.main()
    assert rc == 1
