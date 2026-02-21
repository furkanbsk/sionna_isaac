# Sionna Project Context

## Project Overview

**Sionna** is an open-source Python library for research on communication systems, developed by NVIDIA. It provides hardware-accelerated differentiable simulations for wireless and optical communication systems.

### Core Components

| Component | Description |
|-----------|-------------|
| **Sionna PHY** | Link-level simulator for wireless and optical communication systems |
| **Sionna SYS** | System-level simulator based on physical-layer abstraction |
| **Sionna RT** | Lightning-fast stand-alone ray tracer for radio propagation modeling |

### Local Extension: Isaac Sim + Sionna CSI Pipeline

This workspace includes a local scaffold (`isaacsim_sionna/`) for integrating NVIDIA Isaac Sim with Sionna RT to collect Channel State Information (CSI) datasets for:
- Human Activity Recognition (CSI-HAR)
- CSI-based localization/tracking

## Project Structure

```
sionna/
├── src/sionna/              # Main Python package (phy, sys, rt)
├── ext/sionna-rt/           # Sionna RT extension sources
├── test/                    # Test suite (unit, integration, codes)
├── tutorials/               # Jupyter notebooks and examples
├── isaacsim_sionna/         # Isaac Sim integration project
│   ├── src/isaacsim_sionna/ # Integration code (bridge, exporters, schemas)
│   ├── configs/             # Runtime YAML configurations
│   ├── scripts/             # Command entrypoints
│   ├── tests/               # Integration/unit tests
│   └── data/                # Scenes and generated artifacts
├── 3d_models/               # 3D model assets
├── Terrains/                # Scene definition files
└── doc/                     # Documentation sources
```

## Building, Running, and Testing

### Prerequisites

- **Python**: 3.10–3.12
- **TensorFlow**: 2.14–2.19 (excluding 2.16, 2.17)
- **Ubuntu 24.04** recommended
- For Isaac Sim integration: `env_isaacsim` conda environment

### Installation

```bash
# Install from PyPI
pip install sionna

# Install Sionna RT only
pip install sionna-rt

# Install without RT package
pip install sionna-no-rt

# Install from source (with submodules)
git clone --recursive https://github.com/NVlabs/sionna
pip install ext/sionna-rt/ .
pip install .
```

### Testing

```bash
# Install test requirements
pip install '.[test]'

# Run test suite from repository root
pytest -q test

# Run Isaac Sim integration tests
PYTHONPATH=isaacsim_sionna/src pytest -q isaacsim_sionna/tests

# Run tests on specific GPU
pytest --gpu 1 -q test

# Run tests on CPU
pytest --cpu -q test
```

### Isaac Sim Pipeline Commands

```bash
# Verify Isaac/Sionna dependencies
python isaacsim_sionna/scripts/validate_env.py

# Run pipeline (smoke test)
python isaacsim_sionna/scripts/run_pipeline.py --config isaacsim_sionna/configs/hospital_static.yaml --max-frames 1

# Compare geometry modes
python isaacsim_sionna/scripts/smoke_compare_geometry_modes.py --config isaacsim_sionna/configs/hospital_static.yaml

# Check determinism
python isaacsim_sionna/scripts/smoke_determinism_check.py --config isaacsim_sionna/configs/hospital_static.yaml --max-frames 1
```

### Development Commands

```bash
# Install development requirements
pip install '.[dev]'

# Lint the codebase
pylint src/

# Quick syntax check
python -m py_compile <files>

# Build documentation
pip install '.[doc]'
cd doc && make html
python -m http.server --dir build/html
```

## Coding Conventions

### Style Guidelines

- **Indentation**: 4 spaces
- **Naming**:
  - Functions/files: `snake_case`
  - Classes: `CamelCase`
  - Constants: `UPPER_CASE`
- **Line length**: 80 characters maximum
- **Docstrings**: Required for all public modules, functions, classes (except test methods)

### Module Organization

- Keep modules small and focused (separate bridge, exporter, schema responsibilities)
- Prefer explicit config keys over hardcoded constants
- Avoid committing generated files (respect `.gitignore`)

### Testing Practices

- **Framework**: pytest
- **Test file naming**: `test_*.py`
- **Test function naming**: `test_*`
- **Coverage**:
  - Unit tests for new logic
  - Smoke/integration tests for pipeline changes
- **Validation**:
  - Fast unit behavior (pure Python)
  - Isaac-backed smoke runs (headless, minimal frames)

### Commit Guidelines

- **Format**: Imperative, concise subject line (e.g., `Add hospital USD runtime bridge`)
- **Scope**: Keep commits focused; avoid mixing unrelated changes
- **Sign-off**: Required for all contributions (DCO)
  ```bash
  git commit -s -m "Your commit message"
  ```

### Pull Request Requirements

- Description of what changed and why
- Commands used for validation
- Configs/scenes touched
- Output evidence for pipeline changes (log snippets, schema diffs)

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, build configuration |
| `pylintrc` | Linting rules (Google Python style guide based) |
| `test/conftest.py` | Pytest fixtures and options (GPU/CPU, seeding) |
| `isaacsim_sionna/configs/*.yaml` | Pipeline runtime configurations |

## Key Technical Details

### Sionna RT API (Current)

- `RadioMapSolver` - Radio map computation
- `PathSolver` - Path computation for channel modeling
- `paths.cir(...)` - Channel impulse response extraction

### Isaac Sim Integration Architecture

1. **Isaac Adapter**: Read actor transforms from USD/Isaac world
2. **Scene Translator**: Map static/dynamic geometry between Isaac and Sionna
3. **Sionna Channel Engine**: Compute paths and convert to CSI
4. **Dataset Writer**: Persist CSI, labels, and run metadata

### Data Schema (Per Frame)

- `timestamp_sim`
- `tx_pose`, `rx_pose`
- `actor_poses` (tracked subset)
- `cir_coeff` (`a`) and `delays` (`tau`) or derived CSI tensor
- `label_activity` (HAR) and/or `label_position` (localization)
- `scene_id`, `episode_id`, `frame_id`, `seed`

## Security Notes

- Do not hardcode secrets or tokens in configs/scripts
- Treat large generated outputs as artifacts, not source
- For Isaac runs, prefer headless mode and minimal frame counts during development

## References

- [Official Documentation](https://nvlabs.github.io/sionna/)
- [Sionna RT Repository](https://github.com/NVlabs/sionna-rt)
- [TensorFlow GPU Support](https://www.tensorflow.org/install/gpu)
- [Mitsuba 3 Installation Guide](https://mitsuba.readthedocs.io/en/stable/)
