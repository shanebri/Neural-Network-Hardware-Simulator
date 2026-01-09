from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

# load and validate hardware specs from config/hardware_specs.json
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "hardware_specs.json"


def _load_specs() -> Dict[str, Any]:
    # keep encoding ascii to avoid surprises in the config file
    with _CONFIG_PATH.open("r", encoding="ascii") as handle:
        return json.load(handle)


def _validate_cpu_spec(key: str, spec: Dict[str, Any]) -> None:
    # ensure required fields exist and have sensible ranges
    required = ["name", "frequency", "flops_per_cycle", "core_count"]
    missing = [field for field in required if field not in spec]
    if missing:
        raise ValueError(f"CPU spec '{key}' missing fields: {missing}")
    if spec["frequency"] <= 0 or spec["flops_per_cycle"] <= 0 or spec["core_count"] <= 0:
        raise ValueError(f"CPU spec '{key}' must have positive frequency, flops_per_cycle, and core_count.")
    efficiency = spec.get("efficiency")
    if efficiency is not None and not (0.0 < efficiency <= 1.0):
        raise ValueError(f"CPU spec '{key}' efficiency must be in (0, 1].")


def _validate_gpu_spec(key: str, spec: Dict[str, Any]) -> None:
    # validate required fields and per-precision tflops
    if "name" not in spec:
        raise ValueError(f"GPU spec '{key}' missing 'name'.")
    if "theoretical_tflops" not in spec:
        raise ValueError(f"GPU spec '{key}' missing 'theoretical_tflops'.")
    for precision in ("fp16", "fp32", "fp64"):
        if precision not in spec["theoretical_tflops"]:
            raise ValueError(f"GPU spec '{key}' missing theoretical_tflops['{precision}'].")
        if spec["theoretical_tflops"][precision] <= 0:
            raise ValueError(f"GPU spec '{key}' theoretical_tflops['{precision}'] must be positive.")
    efficiency = spec.get("efficiency", {})
    if efficiency:
        for precision, value in efficiency.items():
            if not (0.0 < value <= 1.0):
                raise ValueError(f"GPU spec '{key}' efficiency['{precision}'] must be in (0, 1].")
    allocation = spec.get("allocation")
    if allocation is not None and not (0.0 < allocation <= 1.0):
        raise ValueError(f"GPU spec '{key}' allocation must be in (0, 1].")


def _validate_specs(specs: Dict[str, Any]) -> None:
    # sanity-check the overall schema and defaults
    defaults = specs.get("defaults", {})
    cpus = specs.get("cpus", {})
    gpus = specs.get("gpus", {})
    if not defaults:
        raise ValueError("hardware_specs.json missing 'defaults'.")
    for key in ("cpu", "gpu", "m4"):
        if key not in defaults:
            raise ValueError(f"hardware_specs.json defaults missing '{key}'.")
    if not cpus or not gpus:
        raise ValueError("hardware_specs.json must define non-empty 'cpus' and 'gpus'.")
    for key, spec in cpus.items():
        _validate_cpu_spec(key, spec)
    for key, spec in gpus.items():
        _validate_gpu_spec(key, spec)
    if defaults["cpu"] not in cpus:
        raise ValueError(f"Default CPU '{defaults['cpu']}' not found in cpus.")
    if defaults["gpu"] not in gpus:
        raise ValueError(f"Default GPU '{defaults['gpu']}' not found in gpus.")
    if defaults["m4"] not in cpus:
        raise ValueError(f"Default M4 '{defaults['m4']}' not found in cpus.")


# load once at import time so callers can access defaults quickly
_SPECS = _load_specs()
_validate_specs(_SPECS)
_DEFAULTS = _SPECS.get("defaults", {})
_CPUS = _SPECS.get("cpus", {})
_GPUS = _SPECS.get("gpus", {})


def list_cpu_keys() -> list[str]:
    # sorted keys make UI dropdowns stable
    return sorted(_CPUS.keys())


def list_gpu_keys() -> list[str]:
    # sorted keys make UI dropdowns stable
    return sorted(_GPUS.keys())


def get_cpu_spec(key: str) -> Dict[str, Any]:
    # raise helpful error if requested key does not exist
    if key not in _CPUS:
        raise KeyError(f"Unknown CPU key '{key}'. Available: {list_cpu_keys()}")
    return _CPUS[key]


def get_gpu_spec(key: str) -> Dict[str, Any]:
    # raise helpful error if requested key does not exist
    if key not in _GPUS:
        raise KeyError(f"Unknown GPU key '{key}'. Available: {list_gpu_keys()}")
    return _GPUS[key]


def get_default_cpu_spec() -> Dict[str, Any]:
    # pull the default cpu key from the config
    return get_cpu_spec(_DEFAULTS.get("cpu", ""))


def get_default_gpu_spec() -> Dict[str, Any]:
    # pull the default gpu key from the config
    return get_gpu_spec(_DEFAULTS.get("gpu", ""))


def get_default_m4_spec() -> Dict[str, Any]:
    # pull the default apple silicon key from the config
    return get_cpu_spec(_DEFAULTS.get("m4", ""))


# module-level defaults used across the project
cpu_specs = get_default_cpu_spec()
gpu_specs = get_default_gpu_spec()
m4_pro_specs = get_default_m4_spec()
