# Roadmap: Isaac Sim -> Sionna RT CSI Dataset Pipeline

## 1. Target Outcomes
1. Generate synchronized tuples: `(scene_state, tx/rx poses, channel params, CSI tensor, labels)`
2. Produce two dataset families:
   1. HAR: labeled motion/gesture/activity windows
   2. Localization: user/target position labels and trajectories
3. Keep runs reproducible with scenario seeds and versioned metadata.

## 2. Key Requirements
1. Isaac Sim environment available in `env_isaacsim`
2. Sionna RT available in same runtime (or bridge runtime with strict sync)
3. Deterministic scenario playback and timestamp alignment
4. Configurable radio setup (carrier, bandwidth, arrays, tx/rx layout)
5. Storage format for large tensors (HDF5/Zarr/NPZ + JSON metadata)

## 3. Technical Architecture
1. **Isaac Adapter**
   1. Read actor transforms from USD/Isaac world
   2. Identify entities mapped to TX/RX or blockers/reflectors
2. **Scene Translator**
   1. Keep static geometry mapping between Isaac and Sionna scene
   2. Update dynamic object transforms each simulation tick
3. **Sionna Channel Engine**
   1. Compute paths periodically (`PathSolver`)
   2. Convert to CSI (`paths.cir(...)` then OFDM/subcarrier mapping)
4. **Dataset Writer**
   1. Persist CSI and labels with timestamps
   2. Write run manifest, config hash, seed, versions

## 4. Milestones
1. M0: Skeleton (current)
   1. Directory layout, configs, placeholder loop
2. M1: Environment and contracts
   1. Validate Isaac/Sionna imports inside `env_isaacsim`
   2. Define schema for one sample frame
3. M2: Static scene end-to-end
   1. Fixed TX/RX in one scene
   2. Export CSI + metadata for N frames
4. M3: Dynamic human motion (HAR)
   1. Add humanoid/animation motion clips
   2. Windowed labels and activity splits
5. M4: Localization dataset
   1. Trajectory ground-truth and multi-receiver setup
   2. Position/velocity labels and benchmark splits
6. M5: Performance and scale
   1. Batch job runner, shard outputs, resumable runs
   2. Optional multi-GPU/offline generation strategy

## 4.1 Progress Snapshot
1. M0 complete
2. M1 complete
3. M2 baseline complete:
   1. Isaac autogen stage fallback with tracked prim poses
   2. Single-path LOS CIR/CSI output in JSONL for smoke runs
4. Next focus: M2.1 scene translator to replace LOS baseline with geometric paths

## 5. Data Schema (first draft)
Per frame:
1. `timestamp_sim`
2. `tx_pose`, `rx_pose`
3. `actor_poses` (tracked subset)
4. `cir_coeff` (`a`) and `delays` (`tau`) or derived CSI tensor
5. `label_activity` (HAR) and/or `label_position` (localization)
6. `scene_id`, `episode_id`, `frame_id`, `seed`

## 6. Risks and Mitigations
1. API drift between Isaac versions:
   1. Pin version in run metadata
2. Real-time performance bottlenecks:
   1. Decimate channel updates (e.g., 5-20 Hz)
   2. Separate physics FPS and radio solve FPS
3. Geometry mismatch between USD and Sionna scenes:
   1. Start with static validation scene and assert transform parity

## 7. References (current APIs)
1. Sionna RT scene/path/radio-map APIs and tutorials
2. Isaac Sim standalone Python workflow
3. Isaac Sim RTX sensor docs for baseline realtime expectations
