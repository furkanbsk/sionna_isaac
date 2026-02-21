"""Viewport debug renderer for Sionna path polylines."""

from __future__ import annotations

from typing import Any


class RayVisualizer:
    """Draw transient path lines in Isaac Sim viewport via DebugDraw."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        isaac_cfg = config.get("isaac", {}) if isinstance(config, dict) else {}
        ray_cfg = isaac_cfg.get("ray_viz", {}) if isinstance(isaac_cfg, dict) else {}

        self.enabled = bool(isaac_cfg.get("visualize_rays", False))
        self.los_color = tuple(float(v) for v in ray_cfg.get("los_color_rgba", [1.0, 0.1, 0.1, 1.0]))
        self.nlos_color = tuple(float(v) for v in ray_cfg.get("nlos_color_rgba", [0.2, 0.7, 1.0, 1.0]))
        self.line_width = max(1, int(round(float(ray_cfg.get("line_width", 2.0)))))

        self._draw = None
        self._initialized = False
        self._warned = False

    def _warn_once(self, message: str) -> None:
        if self._warned:
            return
        self._warned = True
        print(message)

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
            self._warn_once(f"[RayVisualizer] WARN: debug draw unavailable: {exc}")
            self._draw = None
            self._initialized = False
            return False

    def clear(self) -> None:
        if self._draw is None:
            return
        try:
            self._draw.clear_lines()
        except Exception as exc:
            self._warn_once(f"[RayVisualizer] WARN: clear_lines failed: {exc}")

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

            color = self.los_color if bool(path.get("is_los", False)) else self.nlos_color
            for i in range(len(points) - 1):
                p0 = points[i]
                p1 = points[i + 1]
                if not isinstance(p0, list) or not isinstance(p1, list) or len(p0) != 3 or len(p1) != 3:
                    continue
                starts.append((float(p0[0]), float(p0[1]), float(p0[2])))
                ends.append((float(p1[0]), float(p1[1]), float(p1[2])))
                colors.append(color)
                widths.append(self.line_width)

        if not starts:
            return
        try:
            self._draw.draw_lines(starts, ends, colors, widths)
        except Exception as exc:
            self._warn_once(f"[RayVisualizer] WARN: draw_lines failed: {exc}")

    def close(self) -> None:
        self.clear()
        self._draw = None
        self._initialized = False

