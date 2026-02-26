from __future__ import annotations

from isaacsim_sionna.bridge.actor_motion_manager import ActorMotionBackendManager


def test_procedural_backend_routes_to_legacy_controller() -> None:
    cfg = {
        "enabled": True,
        "backend": "procedural",
        "actors": [
            {
                "prim_path": "/World/h0",
                "motion_type": "manual",
                "manual": {"fixed_pose": {"pos_xyz": [1, 2, 3], "quat_wxyz": [1, 0, 0, 0]}},
            }
        ],
    }
    mgr = ActorMotionBackendManager(cfg, root_config={})

    poses = mgr.target_poses(timestamp_sim=0.0, frame_idx=0)
    assert len(poses) == 1
    assert poses[0].prim_path == "/World/h0"
    assert mgr.motion_type_for("/World/h0") == "manual"
    assert mgr.is_ira_backend() is False


def test_ira_backend_uses_tracked_paths() -> None:
    cfg = {
        "enabled": True,
        "backend": "ira",
        "ira": {
            "enabled": True,
            "character": {
                "tracked_prim_paths": ["/World/Characters/Character", "/World/Characters/Character_01"],
            },
        },
    }
    mgr = ActorMotionBackendManager(cfg, root_config={})

    assert mgr.is_ira_backend() is True
    paths = mgr.configured_actor_paths()
    assert "/World/Characters/Character" in paths
    assert mgr.motion_type_for("/World/Characters/Character") == "ira"
