#!/usr/bin/env python3
"""Validate local runtime prerequisites for Isaac Sim + Sionna integration."""

import os
import sys


def check_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def main() -> int:
    print(f"python_executable={sys.executable}")
    print(f"conda_default_env={os.environ.get('CONDA_DEFAULT_ENV', '<none>')}")

    wanted_env = "env_isaacsim"
    if os.environ.get("CONDA_DEFAULT_ENV") != wanted_env:
        print(f"WARNING: expected conda env '{wanted_env}'")

    checks = {
        "sionna.rt": check_module("sionna.rt"),
        "isaacsim": check_module("isaacsim"),
        "isaacsim.simulation_app": check_module("isaacsim.simulation_app"),
        "yaml": check_module("yaml"),
    }

    for k, ok in checks.items():
        print(f"{k}={'OK' if ok else 'MISSING'}")

    required = ["sionna.rt", "isaacsim", "isaacsim.simulation_app", "yaml"]
    missing_required = [k for k in required if not checks[k]]
    if missing_required:
        print(f"ERROR: missing required modules: {missing_required}")
        return 1

    print("NOTE: isaacsim.core.api is loaded after SimulationApp startup in runtime.")

    print("Validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
