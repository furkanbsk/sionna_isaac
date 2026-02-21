"""Pipeline orchestrator scaffold.

Flow:
1. Pull simulation state from Isaac adapter
2. Push state into Sionna adapter
3. Compute channel snapshot
4. Persist sample via exporter
"""

from __future__ import annotations

from datetime import datetime, timezone
import statistics
import sys
import time
import traceback

from isaacsim_sionna.bridge.isaac_adapter import IsaacAdapter
from isaacsim_sionna.exporters.csi_writer import CsiWriter
from isaacsim_sionna.utils.reproducibility import canonicalize_config, compute_config_hash, seed_everything
from isaacsim_sionna.utils.run_manifest import collect_git_info


class Pipeline:
    """End-to-end orchestration for CSI dataset generation."""

    def __init__(self, config: dict):
        self.config = config
        self.isaac = IsaacAdapter(config)
        self.sionna = None
        self.writer = CsiWriter(config)

    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "count": 0}
        vals_ms = [v * 1000.0 for v in values]
        vals_sorted = sorted(vals_ms)
        p50 = vals_sorted[int(0.5 * (len(vals_sorted) - 1))]
        p95 = vals_sorted[int(0.95 * (len(vals_sorted) - 1))]
        return {
            "mean_ms": float(statistics.fmean(vals_ms)),
            "p50_ms": float(p50),
            "p95_ms": float(p95),
            "count": int(len(vals_ms)),
        }

    def run(self) -> None:
        project = self.config.get("project", {})
        seed = int(project.get("seed", 42))
        hash_algo = str(project.get("hash_algo", "sha256")).lower()
        seeded_libraries = seed_everything(seed)
        config_hash = compute_config_hash(self.config, algo=hash_algo)
        timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
        git_info = collect_git_info()
        run_context = {
            "timestamp_utc": timestamp_utc,
            "project_name": str(project.get("name", "isaacsim_sionna")),
            "seed": seed,
            "hash_algo": hash_algo,
            "config_hash": config_hash,
            "git": git_info,
            "runtime": {
                "seeded_libraries": seeded_libraries,
            },
            "config": canonicalize_config(self.config),
        }

        runtime = self.config.get("runtime", {})
        max_frames = int(runtime.get("max_frames", 100))
        isaac_fps = float(runtime.get("isaac_fps", 60))
        radio_hz = float(runtime.get("radio_update_rate_hz", runtime.get("radio_update_hz", 10)))
        frame_skip_ratio = runtime.get("frame_skip_ratio")
        if frame_skip_ratio is not None:
            radio_period = max(int(frame_skip_ratio), 1)
        else:
            radio_period = max(int(round(isaac_fps / radio_hz)), 1)

        t_run_start = time.perf_counter()
        isaac_step_times: list[float] = []
        path_solver_times: list[float] = []
        rgb_render_times: list[float] = []
        hdf5_write_times: list[float] = []
        num_radio_updates = 0

        try:
            self.isaac.start()

            from isaacsim_sionna.bridge.sionna_adapter import (  # pylint: disable=import-outside-toplevel
                SionnaAdapter,
            )

            self.sionna = SionnaAdapter(self.config)

            geometry_proxy = self.isaac.get_sionna_geometry_proxy()
            self.sionna.initialize(geometry_proxy=geometry_proxy)
            self.writer.open(run_context=run_context)

            print(
                f"[Pipeline] start max_frames={max_frames} isaac_fps={isaac_fps} "
                f"radio_update_rate_hz={radio_hz} frame_skip_ratio={frame_skip_ratio} radio_period={radio_period}"
            )

            for frame_idx in range(max_frames):
                t0 = time.perf_counter()
                self.isaac.step()
                isaac_step_times.append(time.perf_counter() - t0)
                state = self.isaac.get_state()

                if frame_idx % radio_period == 0:
                    self.sionna.update_dynamic_state(state)
                    t1 = time.perf_counter()
                    snapshot = self.sionna.compute_snapshot(frame_idx=frame_idx)
                    path_solver_times.append(time.perf_counter() - t1)
                    snapshot["timestamp_sim"] = state.get("timestamp_sim")
                    if hasattr(self.isaac, "ray_visualization_enabled") and self.isaac.ray_visualization_enabled():
                        path_geometry = (
                            self.sionna.get_last_path_geometry() if hasattr(self.sionna, "get_last_path_geometry") else []
                        )
                        if hasattr(self.isaac, "render_paths"):
                            self.isaac.render_paths(path_geometry)
                    render_ref = None
                    if hasattr(self.isaac, "capture_rgb"):
                        tcam = time.perf_counter()
                        render_ref = self.isaac.capture_rgb(frame_idx=frame_idx)
                        rgb_render_times.append(time.perf_counter() - tcam)
                    t2 = time.perf_counter()
                    try:
                        self.writer.write(frame_idx=frame_idx, state=state, snapshot=snapshot, render_ref=render_ref)
                    except TypeError:
                        self.writer.write(frame_idx=frame_idx, state=state, snapshot=snapshot)
                    hdf5_write_times.append(time.perf_counter() - t2)
                    num_radio_updates += 1

                # Real implementation should sync with Isaac's own step clock.
                time.sleep(0.0)

            total_runtime_s = time.perf_counter() - t_run_start
            effective_radio_hz = float(num_radio_updates / max(total_runtime_s, 1e-9))

            geometry_prep_ms = 0.0
            if self.sionna is not None and hasattr(self.sionna, "get_init_metrics"):
                geometry_prep_ms = float((self.sionna.get_init_metrics() or {}).get("geometry_prep_ms", 0.0))

            perf = {
                "isaac_step": self._stats(isaac_step_times),
                "path_solver": self._stats(path_solver_times),
                "rgb_render": self._stats(rgb_render_times),
                "hdf5_write": self._stats(hdf5_write_times),
                "geometry_prep": {
                    "mean_ms": geometry_prep_ms,
                    "p50_ms": geometry_prep_ms,
                    "p95_ms": geometry_prep_ms,
                    "count": 1 if geometry_prep_ms > 0.0 else 0,
                },
                "total_runtime_s": float(total_runtime_s),
                "num_radio_updates": int(num_radio_updates),
                "effective_radio_hz": effective_radio_hz,
            }
            self.writer.set_runtime_metrics(perf)
            print(f"[Pipeline] perf={perf}")
            print("[Pipeline] done")
        except Exception as exc:  # pragma: no cover - runtime failure diagnostics
            print(f"[Pipeline] ERROR: {exc!r}")
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            raise
        finally:
            try:
                self.writer.close()
            finally:
                self.isaac.stop()
            sys.stdout.flush()
            sys.stderr.flush()
