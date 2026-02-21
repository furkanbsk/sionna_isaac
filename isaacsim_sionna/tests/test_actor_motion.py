from __future__ import annotations

import math

from isaacsim_sionna.bridge.actor_motion import ActorMotionManager, angular_velocity_from_quats, yaw_to_quat_wxyz


def test_waypoint_trajectory_progresses_deterministically() -> None:
    cfg = {
        "enabled": True,
        "actors": [
            {
                "prim_path": "/World/humanoid_01",
                "motion_type": "trajectory",
                "trajectory": {
                    "kind": "waypoints",
                    "loop": False,
                    "speed_mps": 1.0,
                    "waypoints_xyz": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                },
            }
        ],
    }
    manager = ActorMotionManager(cfg)

    poses_t0 = manager.target_poses(timestamp_sim=0.0, frame_idx=0)
    poses_t1 = manager.target_poses(timestamp_sim=1.0, frame_idx=60)

    assert poses_t0[0].position_xyz == [0.0, 0.0, 0.0]
    assert poses_t1[0].position_xyz == [1.0, 0.0, 0.0]


def test_angular_velocity_from_quaternion_delta() -> None:
    q0 = yaw_to_quat_wxyz(0.0)
    q1 = yaw_to_quat_wxyz(math.pi / 2.0)

    omega = angular_velocity_from_quats(q0, q1, dt=1.0)
    assert abs(omega[0]) < 1e-6
    assert abs(omega[1]) < 1e-6
    assert abs(omega[2] - (math.pi / 2.0)) < 1e-6
