"""Pipeline orchestrator scaffold.

Flow:
1. Pull simulation state from Isaac adapter
2. Push state into Sionna adapter
3. Compute channel snapshot
4. Persist sample via exporter
"""

from __future__ import annotations

import time

from isaacsim_sionna.bridge.isaac_adapter import IsaacAdapter
from isaacsim_sionna.bridge.sionna_adapter import SionnaAdapter
from isaacsim_sionna.exporters.csi_writer import CsiWriter


class Pipeline:
    """End-to-end orchestration for CSI dataset generation."""

    def __init__(self, config: dict):
        self.config = config
        self.isaac = IsaacAdapter(config)
        self.sionna = SionnaAdapter(config)
        self.writer = CsiWriter(config)

    def run(self) -> None:
        runtime = self.config.get("runtime", {})
        max_frames = int(runtime.get("max_frames", 100))
        isaac_fps = float(runtime.get("isaac_fps", 60))
        radio_hz = float(runtime.get("radio_update_hz", 10))
        radio_period = max(int(round(isaac_fps / radio_hz)), 1)

        self.isaac.start()
        try:
            geometry_proxy = self.isaac.get_sionna_geometry_proxy()
            self.sionna.initialize(geometry_proxy=geometry_proxy)
            self.writer.open()

            print(
                f"[Pipeline] start max_frames={max_frames} isaac_fps={isaac_fps} "
                f"radio_update_hz={radio_hz} radio_period={radio_period}"
            )

            for frame_idx in range(max_frames):
                self.isaac.step()
                state = self.isaac.get_state()

                if frame_idx % radio_period == 0:
                    self.sionna.update_dynamic_state(state)
                    snapshot = self.sionna.compute_snapshot()
                    self.writer.write(frame_idx=frame_idx, state=state, snapshot=snapshot)

                # Real implementation should sync with Isaac's own step clock.
                time.sleep(0.0)

            print("[Pipeline] done")
        finally:
            self.writer.close()
            self.isaac.stop()
