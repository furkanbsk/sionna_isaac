from __future__ import annotations

from pathlib import Path

import yaml

from isaacsim_sionna.bridge.ira_motion import IraMotionBackend


def test_runtime_config_and_commands_generated(tmp_path: Path) -> None:
    cfg = {
        "enabled": True,
        "backend": "ira",
        "ira": {
            "enabled": True,
            "runtime_dir": str(tmp_path),
            "auto_generate_config": True,
            "auto_generate_commands": True,
            "character": {
                "num": 2,
            },
        },
    }
    backend = IraMotionBackend(cfg, root_config={})

    paths = backend._build_runtime_files(scene_usd="/tmp/scene.usd", seed=7, max_frames=12)  # pylint: disable=protected-access

    assert paths.config_yaml.exists()
    assert paths.command_txt.exists()
    assert paths.robot_command_txt.exists()

    payload = yaml.safe_load(paths.config_yaml.read_text(encoding="utf-8"))
    ira = payload["isaacsim.replicator.agent"]
    assert ira["global"]["seed"] == 7
    assert ira["scene"]["asset_path"] == "/tmp/scene.usd"
    assert ira["character"]["num"] == 2

    lines = [line.strip() for line in paths.command_txt.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any("Character Idle" in line for line in lines)
    assert any("GoTo" in line for line in lines)
