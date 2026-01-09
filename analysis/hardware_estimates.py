def compute_flop_rate_cpu(specs, efficiency=None, allocation=1.0):
    """
    Compute a practical FLOP rate for a CPU.

    Args:
        specs: dict with keys frequency (Hz), flops_per_cycle (per core), core_count.
        efficiency: optional sustained efficiency (0-1). Defaults to specs["efficiency"].
        allocation: fraction of the device available to this workload (0-1).

    Returns:
        Effective FLOPs/second.
    """
    # fall back to spec defaults when the caller does not override them
    sustained_eff = efficiency if efficiency is not None else specs.get("efficiency", 1.0)
    core_count = specs.get("core_count", 1)
    # basic throughput model: frequency * ops per cycle * cores * efficiency * allocation
    return specs["frequency"] * specs["flops_per_cycle"] * core_count * sustained_eff * allocation


def compute_flop_rate_gpu(specs, precision="fp32", efficiency=None, allocation=None):
    """
    Compute an effective FLOP rate for a GPU at a given precision.

    Args:
        specs: dict with key "theoretical_tflops" mapping precision -> TFLOP/s.
        precision: one of the keys in specs["theoretical_tflops"] (e.g., "fp16", "fp32", "fp64").
        efficiency: optional sustained efficiency (0-1). Defaults to specs["efficiency"][precision] or specs["default_efficiency"].
        allocation: fraction of GPU time/SMs available to this workload (0-1). Defaults to specs["allocation"] or 1.0.
    """
    # convert TFLOP/s to FLOP/s then scale by efficiency and allocation
    theoretical = specs["theoretical_tflops"][precision] * 1e12
    default_eff = specs.get("efficiency", {}).get(precision, specs.get("default_efficiency", 1.0))
    sustained_eff = efficiency if efficiency is not None else default_eff
    alloc = allocation if allocation is not None else specs.get("allocation", 1.0)
    return theoretical * sustained_eff * alloc


def estimate_runtime(total_ops, flop_rate):
    """
    Estimate runtime in seconds given a total operation count and FLOP rate.
    """
    # simple throughput model: seconds = ops / ops_per_second
    return total_ops / flop_rate
