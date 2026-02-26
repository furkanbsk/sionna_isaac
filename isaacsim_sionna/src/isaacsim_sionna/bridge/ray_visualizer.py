"""Viewport debug renderer for Sionna path polylines."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RayVisualizer:
    """Draw transient path lines in Isaac Sim viewport via DebugDraw."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        isaac_cfg = config.get("isaac", {}) if isinstance(config, dict) else {}
        ray_cfg = isaac_cfg.get("ray_viz", {}) if isinstance(isaac_cfg, dict) else {}

        self.enabled = bool(isaac_cfg.get("visualize_rays", False))
        self.los_color = self._validate_rgba(ray_cfg.get("los_color_rgba", [1.0, 0.1, 0.1, 1.0]))
        self.nlos_color = self._validate_rgba(ray_cfg.get("nlos_color_rgba", [0.2, 0.7, 1.0, 1.0]))
        self.specular_color = self._validate_rgba(ray_cfg.get("specular_color_rgba", [0.45, 0.65, 1.0, 1.0]))
        self.diffuse_color = self._validate_rgba(ray_cfg.get("diffuse_color_rgba", [0.45, 1.0, 0.55, 1.0]))
        self.refraction_color = self._validate_rgba(ray_cfg.get("refraction_color_rgba", [1.0, 0.55, 0.35, 1.0]))
        self.diffraction_color = self._validate_rgba(ray_cfg.get("diffraction_color_rgba", [0.85, 0.3, 1.0, 1.0]))
        self.color_by_interaction = bool(ray_cfg.get("color_by_interaction", True))
        self.line_width = max(1, int(round(float(ray_cfg.get("line_width", 2.0)))))

        self._draw = None
        self._initialized = False
        self._warned_keys: set[str] = set()

    @staticmethod
    def _validate_rgba(value: Any) -> tuple[float, float, float, float]:
        """Validate and normalize an RGBA color tuple."""
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            logger.warning("Invalid RGBA color %r, using default (1,1,1,1)", value)
            return (1.0, 1.0, 1.0, 1.0)
        return tuple(float(v) for v in value)  # type: ignore[return-value]

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message)

    def initialize(self) -> bool:
        if not self.enabled:
            return False
        if self._initialized:
            return True

        try:
            from isaacsim.util.debug_draw import _debug_draw  # pylint: disable=import-outside-toplevel

            self._draw = _debug_draw.acquire_debug_draw_interface()
            self._initialized = True
            return True
        except Exception as exc:
            self._warn_once("debug_draw_init", f"[RayVisualizer] WARN: debug draw unavailable: {exc}")
            self._draw = None
            self._initialized = False
            return False

    def clear(self) -> None:
        if self._draw is None:
            return
        try:
            self._draw.clear_lines()
        except Exception as exc:
            self._warn_once("clear_lines", f"[RayVisualizer] WARN: clear_lines failed: {exc}")

    def draw_paths(self, path_geometry: list[dict[str, Any]] | None) -> None:
        if not self._initialized or self._draw is None:
            return

        self.clear()
        if not path_geometry:
            return

        starts = []
        ends = []
        colors = []
        widths = []

        for path in path_geometry:
            points = path.get("points_xyz") if isinstance(path, dict) else None
            if not isinstance(points, list) or len(points) < 2:
                continue

            interaction_types = path.get("interaction_types") if isinstance(path, dict) else None
            color = self.los_color if bool(path.get("is_los", False)) else self.nlos_color
            for i in range(len(points) - 1):
                p0 = points[i]
                p1 = points[i + 1]
                if not isinstance(p0, list) or not isinstance(p1, list) or len(p0) != 3 or len(p1) != 3:
                    continue
                color_i = color
                if self.color_by_interaction and isinstance(interaction_types, list):
                    if i < len(interaction_types):
                        color_i = self._color_for_interaction(int(interaction_types[i]))
                starts.append((float(p0[0]), float(p0[1]), float(p0[2])))
                ends.append((float(p1[0]), float(p1[1]), float(p1[2])))
                colors.append(color_i)
                widths.append(self.line_width)

        if not starts:
            return
        try:
            self._draw.draw_lines(starts, ends, colors, widths)
        except Exception as exc:
            self._warn_once("draw_lines", f"[RayVisualizer] WARN: draw_lines failed: {exc}")

    def close(self) -> None:
        self.clear()
        self._draw = None
        self._initialized = False

    def _color_for_interaction(self, interaction_type: int) -> tuple[float, float, float, float]:
        # Sionna InteractionType bit flags:
        # SPECULAR=1, DIFFUSE=2, REFRACTION=4, DIFFRACTION=8
        if interaction_type & 8:
            return self.diffraction_color
        if interaction_type & 4:
            return self.refraction_color
        if interaction_type & 2:
            return self.diffuse_color
        if interaction_type & 1:
            return self.specular_color
        return self.nlos_color
