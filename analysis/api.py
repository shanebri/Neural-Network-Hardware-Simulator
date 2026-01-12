from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

from analysis.benchmark import MeasurementResult, measure_forward_time
from analysis.flopCounter import count_model_ops, ops_breakdown_to_flops
from analysis.hardware_estimates import (
    compute_flop_rate_cpu,
    compute_flop_rate_gpu,
    estimate_runtime,
)
from models.hardware_specs import (
    cpu_specs,
    gpu_specs,
    get_cpu_spec,
    get_gpu_spec,
)


# public helpers for estimating and measuring model runtimes
HardwareKey = Literal["cpu", "gpu"]
Precision = Literal["fp16", "fp32", "fp64"]

NORMALIZED_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EstimateResult:
    hardware: str
    precision: str
    batch_size: int
    macs: int
    flops: int
    flop_rate: float
    est_runtime: float
    breakdown: dict
    metadata: dict


def _resolve_hardware_spec(hardware: HardwareKey) -> dict:
    # map the short hardware key to the module defaults
    if hardware == "cpu":
        return cpu_specs
    if hardware == "gpu":
        return gpu_specs
    raise ValueError(f"Unsupported hardware '{hardware}'. Use one of: cpu, gpu.")


def estimate_model_runtime(
    model,
    batch_size: int,
    hardware: HardwareKey = "cpu",
    hardware_profile: str | None = None,
    precision: Precision = "fp32",
    include_bias: bool = True,
    include_activations: bool = False,
    activation_ops_per_element: int = 1,
    fma_cost: int = 2,
    efficiency: float | None = None,
    allocation: float | None = None,
) -> EstimateResult:
    """
    Estimate runtime for a model forward pass on a single hardware target.

    Returns a structured estimate with op counts, FLOP rate, and runtime.
    Set hardware_profile to select a specific entry from config/hardware_specs.json.
    """
    # only gpu estimates allow precision selection
    if hardware != "gpu" and precision != "fp32":
        raise ValueError("Precision is only configurable for GPU estimates.")

    # resolve the hardware spec from defaults or from a specific profile
    if hardware_profile:
        if hardware == "gpu":
            spec = get_gpu_spec(hardware_profile)
        else:
            spec = get_cpu_spec(hardware_profile)
    else:
        spec = _resolve_hardware_spec(hardware)
    total_ops, breakdown = count_model_ops(
        model,
        batch_size=batch_size,
        include_bias=include_bias,
        include_activations=include_activations,
        activation_ops_per_element=activation_ops_per_element,
    )

    # extract counts for reporting
    macs = breakdown.get("macs", 0)
    flops = ops_breakdown_to_flops(breakdown, fma_cost=fma_cost)

    # pick the correct flop-rate model based on hardware
    if hardware == "gpu":
        if precision not in spec["theoretical_tflops"]:
            raise ValueError(
                f"Unsupported precision '{precision}' for GPU. "
                f"Choose from {sorted(spec['theoretical_tflops'].keys())}."
            )
        flop_rate = compute_flop_rate_gpu(
            spec, precision=precision, efficiency=efficiency, allocation=allocation
        )
    else:
        # force fp32 for cpu paths
        precision = "fp32"
        flop_rate = compute_flop_rate_cpu(
            spec, efficiency=efficiency, allocation=allocation or 1.0
        )

    # capture inputs so normalized outputs can be traced back later
    metadata = {
        "model_class": model.__class__.__name__,
        "model_repr": repr(model),
        "hardware_key": hardware,
        "hardware_profile": hardware_profile,
        "include_bias": include_bias,
        "include_activations": include_activations,
        "activation_ops_per_element": activation_ops_per_element,
        "fma_cost": fma_cost,
        "efficiency": efficiency,
        "allocation": allocation,
    }

    return EstimateResult(
        hardware=spec["name"],
        precision=precision,
        batch_size=batch_size,
        macs=macs,
        flops=flops,
        flop_rate=flop_rate,
        est_runtime=estimate_runtime(flops, flop_rate),
        breakdown=breakdown,
        metadata=metadata,
    )


def estimate_runtime_table(
    model,
    batch_sizes: Sequence[int],
    gpu_precisions: Iterable[str] = ("fp16", "fp32", "fp64"),
    include_gpu: bool = True,
    include_cpu: bool = True,
):
    """
    Convenience wrapper returning a tidy DataFrame of runtime estimates.
    """
    # defer import so this module stays lightweight for non-plot use cases
    from analysis.visualization import build_runtime_table

    return build_runtime_table(
        model=model,
        batch_sizes=batch_sizes,
        gpu_precisions=gpu_precisions,
        include_gpu=include_gpu,
        include_cpu=include_cpu,
    )


def measure_model_runtime(
    model,
    input_shape: Sequence[int],
    batch_size: int = 1,
    device: str = "cpu",
    runs: int = 50,
    warmup: int = 10,
    use_inference_mode: bool = True,
) -> MeasurementResult:
    """
    Measure forward-pass runtime (inference) for a model on a device.
    """
    # delegate to the benchmark helper for timing details
    return measure_forward_time(
        model=model,
        input_shape=input_shape,
        batch_size=batch_size,
        device=device,
        runs=runs,
        warmup=warmup,
        use_inference_mode=use_inference_mode,
    )


def compare_estimate_to_measurement(
    estimate: EstimateResult,
    measurement: MeasurementResult,
) -> dict:
    """
    Compare a theoretical estimate to a measured runtime.
    """
    # ratio > 1 means the measurement is slower than the estimate
    ratio = (
        measurement.mean_s / estimate.est_runtime
        if estimate.est_runtime > 0
        else float("inf")
    )
    return {
        "estimated_s": estimate.est_runtime,
        "measured_mean_s": measurement.mean_s,
        "ratio_measured_to_estimated": ratio,
        "measured_samples_per_s": measurement.samples_per_s,
    }


def normalize_estimate_result(estimate: EstimateResult) -> dict:
    """
    Normalize a runtime estimate into a consistent, UI-friendly structure.
    """
    # flatten dataclass content into serializable dicts
    return {
        "kind": "estimate",
        "version": NORMALIZED_SCHEMA_VERSION,
        "model": {
            "class": estimate.metadata.get("model_class"),
            "repr": estimate.metadata.get("model_repr"),
        },
        "hardware": {
            "key": estimate.metadata.get("hardware_key"),
            "profile": estimate.metadata.get("hardware_profile"),
            "name": estimate.hardware,
            "precision": estimate.precision,
        },
        "input": {
            "batch_size": estimate.batch_size,
        },
        "ops": {
            "macs": estimate.macs,
            "flops": estimate.flops,
            "breakdown": estimate.breakdown,
        },
        "runtime": {
            "estimated_s": estimate.est_runtime,
            "flop_rate": estimate.flop_rate,
        },
        "assumptions": {
            "include_bias": estimate.metadata.get("include_bias"),
            "include_activations": estimate.metadata.get("include_activations"),
            "activation_ops_per_element": estimate.metadata.get("activation_ops_per_element"),
            "fma_cost": estimate.metadata.get("fma_cost"),
            "efficiency": estimate.metadata.get("efficiency"),
            "allocation": estimate.metadata.get("allocation"),
        },
    }


def normalize_measurement_result(
    measurement: MeasurementResult,
    model,
    hardware_key: str | None = None,
    hardware_profile: str | None = None,
) -> dict:
    """
    Normalize a runtime measurement into a consistent, UI-friendly structure.
    """
    # include both runtime stats and throughput
    return {
        "kind": "measurement",
        "version": NORMALIZED_SCHEMA_VERSION,
        "model": {
            "class": model.__class__.__name__,
            "repr": repr(model),
        },
        "hardware": {
            "key": hardware_key,
            "profile": hardware_profile,
            "name": measurement.device,
            "precision": None,
        },
        "input": {
            "batch_size": measurement.batch_size,
            "input_shape": list(measurement.input_shape),
        },
        "runtime": {
            "mean_s": measurement.mean_s,
            "median_s": measurement.median_s,
            "stdev_s": measurement.stdev_s,
            "min_s": measurement.min_s,
            "max_s": measurement.max_s,
        },
        "throughput": {
            "samples_per_s": measurement.samples_per_s,
        },
        "measurement": {
            "runs": measurement.runs,
            "warmup": measurement.warmup,
        },
    }


def normalize_comparison_result(
    estimate: EstimateResult,
    measurement: MeasurementResult,
) -> dict:
    """
    Normalize estimate vs measurement comparison into a consistent structure.
    """
    # keep the comparison payload small and predictable
    comparison = compare_estimate_to_measurement(estimate, measurement)
    return {
        "kind": "comparison",
        "version": NORMALIZED_SCHEMA_VERSION,
        "estimated_s": comparison["estimated_s"],
        "measured_mean_s": comparison["measured_mean_s"],
        "ratio_measured_to_estimated": comparison["ratio_measured_to_estimated"],
        "measured_samples_per_s": comparison["measured_samples_per_s"],
    }


def normalized_schemas() -> dict:
    """
    Return JSON-schema-like shapes for UI validation and documentation.
    """
    # simple schema hints for consumers (not strict jsonschema)
    return {
        "version": NORMALIZED_SCHEMA_VERSION,
        "estimate": {
            "kind": "estimate",
            "version": NORMALIZED_SCHEMA_VERSION,
            "model": {"class": "str", "repr": "str"},
            "hardware": {"key": "str", "profile": "str|None", "name": "str", "precision": "str"},
            "input": {"batch_size": "int"},
            "ops": {"macs": "int", "flops": "int", "breakdown": "dict"},
            "runtime": {"estimated_s": "float", "flop_rate": "float"},
            "assumptions": {
                "include_bias": "bool",
                "include_activations": "bool",
                "activation_ops_per_element": "int",
                "fma_cost": "int",
                "efficiency": "float|None",
                "allocation": "float|None",
            },
        },
        "measurement": {
            "kind": "measurement",
            "version": NORMALIZED_SCHEMA_VERSION,
            "model": {"class": "str", "repr": "str"},
            "hardware": {"key": "str|None", "profile": "str|None", "name": "str", "precision": "None"},
            "input": {"batch_size": "int", "input_shape": "list[int]"},
            "runtime": {
                "mean_s": "float",
                "median_s": "float",
                "stdev_s": "float",
                "min_s": "float",
                "max_s": "float",
            },
            "throughput": {"samples_per_s": "float"},
            "measurement": {"runs": "int", "warmup": "int"},
        },
        "comparison": {
            "kind": "comparison",
            "version": NORMALIZED_SCHEMA_VERSION,
            "estimated_s": "float",
            "measured_mean_s": "float",
            "ratio_measured_to_estimated": "float",
            "measured_samples_per_s": "float",
        },
    }


def run_normalized_pipeline(
    model,
    input_shape: Sequence[int],
    batch_size: int,
    hardware: HardwareKey = "cpu",
    hardware_profile: str | None = None,
    precision: Precision = "fp32",
    device: str = "cpu",
    runs: int = 50,
    warmup: int = 10,
    include_bias: bool = True,
    include_activations: bool = False,
    activation_ops_per_element: int = 1,
    fma_cost: int = 2,
    efficiency: float | None = None,
    allocation: float | None = None,
    use_inference_mode: bool = True,
) -> dict:
    """
    End-to-end helper that returns normalized estimate, measurement, and comparison.
    """
    # compute the estimate first so we can compare later
    estimate = estimate_model_runtime(
        model=model,
        batch_size=batch_size,
        hardware=hardware,
        hardware_profile=hardware_profile,
        precision=precision,
        include_bias=include_bias,
        include_activations=include_activations,
        activation_ops_per_element=activation_ops_per_element,
        fma_cost=fma_cost,
        efficiency=efficiency,
        allocation=allocation,
    )
    measurement = measure_model_runtime(
        model=model,
        input_shape=input_shape,
        batch_size=batch_size,
        device=device,
        runs=runs,
        warmup=warmup,
        use_inference_mode=use_inference_mode,
    )
    # combine the outputs into a single payload
    return {
        "estimate": normalize_estimate_result(estimate),
        "measurement": normalize_measurement_result(
            measurement,
            model=model,
            hardware_key=hardware,
            hardware_profile=hardware_profile,
        ),
        "comparison": normalize_comparison_result(estimate, measurement),
    }
