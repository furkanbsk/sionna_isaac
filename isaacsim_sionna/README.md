# Isaac Sim + Sionna CSI Pipeline

This directory is an empty but structured starting point for integrating **NVIDIA Isaac Sim** and **Sionna RT** to collect CSI datasets.

## Final Goal
Collect reproducible CSI datasets from simulation for:
1. Human Activity Recognition (CSI-HAR)
2. CSI-based localization/tracking

## Current Status
Implemented baseline with real USD geometry proxy:
1. Isaac adapter boot/step/state export
2. Pipeline loop with configurable radio update rate
3. Runtime USD mesh-AABB extraction (hospital scene)
4. Runtime Sionna XML generation and `PathSolver` snapshots
5. JSONL sample writing with CIR/CSI + path count metadata
6. Run-level determinism metadata (`manifest.json`) with config hash and git commit

## Environment
You said Isaac Sim runs in conda env: `env_isaacsim`.

Expected runtime pattern:
1. Activate `env_isaacsim`
2. Run Isaac Sim Python entrypoint from that environment
3. Execute this pipeline

## Quick Start
```bash
cd /home/nvidia/Desktop/Main_Workspace/sionna
python isaacsim_sionna/scripts/validate_env.py
python isaacsim_sionna/scripts/run_pipeline.py --config isaacsim_sionna/configs/hospital_static.yaml --max-frames 1
```

Geometry mode toggle (in YAML):
- `isaac.geometry.mode: mesh` for runtime USD mesh export (PLY -> Sionna XML)
- `isaac.geometry.mode: aabb` for fast fallback proxy mode

Determinism controls (in YAML `project` block):
- `seed`: base seed for Python/NumPy/(optional) Torch/TensorFlow
- `solver_seed_strategy`: `frame_offset` (default) or `fixed`
- `hash_algo`: config hash algorithm (`sha256` default)

Each run writes:
- `samples.jsonl`
- `manifest.json` (timestamp, config hash, seed, git info, merged config, output hash)

Quick comparison:
```bash
python isaacsim_sionna/scripts/smoke_compare_geometry_modes.py --config isaacsim_sionna/configs/hospital_static.yaml
```

Determinism smoke check:
```bash
python isaacsim_sionna/scripts/smoke_determinism_check.py --config isaacsim_sionna/configs/hospital_static.yaml --max-frames 1
```

## Project Layout
- `configs/`: pipeline and scenario configs
- `data/`: raw/interim/processed outputs and metadata
- `docs/ROADMAP.md`: phased plan to reach CSI dataset production
- `scripts/`: command entrypoints
- `src/isaacsim_sionna/`: integration code skeleton
- `tests/`: placeholders for integration/contract tests

## Notes on APIs
Older tutorials may show:
- `scene.coverage_map()`
- `scene.compute_paths()`

Current Sionna RT uses:
- `RadioMapSolver`
- `PathSolver`
- `paths.cir(...)`

See `docs/ROADMAP.md` for integration details and milestones.
