from __future__ import annotations

from isaacsim_sionna.utils.reproducibility import compute_config_hash


def test_config_hash_stable_under_key_order() -> None:
    a = {
        "project": {"seed": 42, "name": "demo"},
        "runtime": {"max_frames": 10, "isaac_fps": 30},
    }
    b = {
        "runtime": {"isaac_fps": 30, "max_frames": 10},
        "project": {"name": "demo", "seed": 42},
    }
    assert compute_config_hash(a) == compute_config_hash(b)


def test_config_hash_changes_when_seed_changes() -> None:
    a = {"project": {"seed": 42}}
    b = {"project": {"seed": 43}}
    assert compute_config_hash(a) != compute_config_hash(b)
