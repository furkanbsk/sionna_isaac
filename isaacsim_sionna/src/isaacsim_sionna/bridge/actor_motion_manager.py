"""Actor motion backend dispatcher.

Supports:
- procedural (legacy waypoint/circle/manual)
- ira (isaacsim.replicator.agent)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ActorMotionBackendManager:
    """Facade for selecting and driving actor motion backend."""

    def __init__(self, actor_motion_cfg: dict[str, Any] | None, root_config: dict[str, Any] | None = None):
        actor_motion_cfg = actor_motion_cfg or {}
        self._cfg = actor_motion_cfg
        self._root = root_config or {}
        self.enabled = bool(actor_motion_cfg.get("enabled", False))
        self.update_every_tick = bool(actor_motion_cfg.get("update_every_tick", True))

        self.backend_name = str(actor_motion_cfg.get("backend", "procedural")).strip().lower() or "procedural"
        self._procedural = None
        self._ira = None

    @property
    def _procedural_backend(self):
        if self._procedural is None:
            from isaacsim_sionna.bridge.actor_motion import ActorMotionManager  # pylint: disable=import-outside-toplevel
            self._procedural = ActorMotionManager(self._cfg)
        return self._procedural

    @property
    def _ira_backend(self):
        if self._ira is None:
            from isaacsim_sionna.bridge.ira_motion import IraMotionBackend  # pylint: disable=import-outside-toplevel
            self._ira = IraMotionBackend(self._cfg, root_config=self._root)
        return self._ira

    def is_ira_backend(self) -> bool:
        return self.backend_name == "ira"

    def start(self, *, scene_usd: str | None, headless: bool, seed: int, max_frames: int) -> bool:
        if not self.enabled:
            return False
        if self.backend_name == "ira":
            return self._ira_backend.start(scene_usd=scene_usd, headless=headless, seed=seed, max_frames=max_frames)
        return True

    def stop(self) -> None:
        if self.backend_name == "ira":
            self._ira_backend.stop()

    @property
    def init_error(self) -> str | None:
        if self.backend_name == "ira":
            return self._ira_backend.init_error
        return None

    def configured_actor_paths(self) -> list[str]:
        if self.backend_name == "ira":
            return self._ira_backend.configured_actor_paths()
        return self._procedural_backend.configured_actor_paths()

    def runtime_actor_paths(self) -> list[str]:
        if self.backend_name == "ira":
            return self._ira_backend.runtime_actor_paths()
        return self._procedural_backend.configured_actor_paths()

    def motion_type_for(self, prim_path: str) -> str:
        if self.backend_name == "ira":
            return self._ira_backend.motion_type_for(prim_path)
        return self._procedural_backend.motion_type_for(prim_path)

    def target_poses(self, timestamp_sim: float, frame_idx: int):
        if not self.enabled:
            return []
        if self.backend_name == "ira":
            return self._ira_backend.target_poses(timestamp_sim, frame_idx)
        return self._procedural_backend.target_poses(timestamp_sim, frame_idx)
