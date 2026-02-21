"""Deterministic run helpers."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _normalize(value: Any) -> Any:
    """Normalize nested structures for stable JSON hashing."""
    if isinstance(value, Mapping):
        return {str(k): _normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def canonicalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic dictionary representation."""
    return _normalize(config)


def stable_json_dumps(obj: Any) -> str:
    """Serialize JSON in a stable way across runs."""
    return json.dumps(_normalize(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_config_hash(config: dict[str, Any], algo: str = "sha256") -> str:
    """Hash active config dictionary."""
    payload = stable_json_dumps(canonicalize_config(config)).encode("utf-8")
    if algo not in hashlib.algorithms_available:
        raise ValueError(f"Unsupported hash algorithm: {algo}")
    digest = hashlib.new(algo)
    digest.update(payload)
    return digest.hexdigest()


def seed_everything(seed: int) -> list[str]:
    """Set deterministic seeds for available runtime libraries."""
    applied = []
    random.seed(seed)
    applied.append("random")

    np.random.seed(seed)
    applied.append("numpy")

    # Do not import heavy frameworks proactively before Isaac SimulationApp starts.
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            tf.random.set_seed(seed)
            applied.append("tensorflow")
        except Exception:
            pass

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            applied.append("torch")
        except Exception:
            pass

    return applied


def seed_isaac_runtime(seed: int) -> list[str]:
    """Best-effort Isaac/Replicator seeding after SimulationApp startup."""
    applied = []

    try:
        import omni.replicator.core as rep  # pylint: disable=import-error

        set_seed = getattr(rep, "set_global_seed", None)
        if callable(set_seed):
            set_seed(seed)
            applied.append("isaacsim_replicator")
    except Exception:
        pass

    try:
        import carb.settings  # pylint: disable=import-error

        settings = carb.settings.get_settings()
        settings.set("/omni/replicator/seed", int(seed))
        applied.append("carb_settings")
    except Exception:
        pass

    return applied
