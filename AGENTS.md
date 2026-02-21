# Repository Guidelines

## Project Structure & Module Organization
- `src/sionna/`: main Sionna Python package.
- `ext/sionna-rt/`: Sionna RT extension sources, docs, and tests.
- `test/`: repository-level tests (including `test/conftest.py`).
- `tutorials/`: notebooks and examples.
- `isaacsim_sionna/`: Isaac Sim + Sionna integration project.
  - `isaacsim_sionna/src/isaacsim_sionna/bridge/`: adapters and pipeline orchestration.
  - `isaacsim_sionna/configs/`: runtime YAML configs.
  - `isaacsim_sionna/scripts/`: runnable entry points.
  - `isaacsim_sionna/tests/`: focused integration/unit tests.
  - `isaacsim_sionna/data/`: scenes and generated artifacts (`raw/`, `scenes_*`).

## Build, Test, and Development Commands
- `python -m py_compile <files>`: quick syntax verification.
- `PYTHONPATH=isaacsim_sionna/src pytest -q isaacsim_sionna/tests`: run integration project tests.
- `pytest -q test`: run repository test suite (from repo root).
- `python isaacsim_sionna/scripts/validate_env.py`: verify Isaac/Sionna dependencies.
- `python isaacsim_sionna/scripts/run_pipeline.py --config isaacsim_sionna/configs/hospital_static.yaml --max-frames 1`: smoke-run pipeline.

Use `env_isaacsim` Python explicitly when needed:
`/home/nvidia/miniconda3/envs/env_isaacsim/bin/python ...`

## Coding Style & Naming Conventions
- Python: 4-space indentation, PEP 8 naming (`snake_case` functions/files, `CamelCase` classes).
- Keep modules small and focused (e.g., bridge, exporter, schema responsibilities separated).
- Prefer explicit config keys over hardcoded constants.
- Avoid committing generated files; respect `.gitignore` in `isaacsim_sionna/`.

## Testing Guidelines
- Framework: `pytest`.
- Test files: `test_*.py`; test functions: `test_*`.
- Add unit tests for new logic and at least one smoke/integration path for pipeline changes.
- For runtime-dependent features, validate both:
  1. Fast unit behavior (pure Python).
  2. Isaac-backed smoke run (headless, short frame count).

## Commit & Pull Request Guidelines
- Commit style: imperative, concise subject (e.g., `Add hospital USD runtime bridge`).
- Keep commits scoped (avoid mixing unrelated workspace changes).
- PRs should include:
  - What changed and why.
  - Commands used to validate.
  - Configs/scenes touched.
  - Output evidence for pipeline changes (log snippet or sample schema diff).

## Security & Configuration Tips
- Do not hardcode secrets or tokens in configs/scripts.
- Treat large generated outputs as artifacts, not source.
- For Isaac runs, prefer headless mode and minimal frame counts during development.
