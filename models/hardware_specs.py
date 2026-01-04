cpu_specs = {
    "name": "Intel i9 9900k",
    "frequency": 3.6e9,
    "flops_per_cycle": 16,  # AVX2 fused-multiply-add per core
    "core_count": 8,
    "efficiency": 0.7  # typical sustained utilization under real workloads
}

gpu_specs = {
    "name": "NVIDIA GeForce RTX 4070 Ti",
    # Published peak throughput per precision in TFLOP/s (non-sparsity, non-tensor)
    "theoretical_tflops": {
        "fp16": 80.0,   # approx 2x fp32 rate
        "fp32": 40.0,
        "fp64": 0.63    # 1/64th fp32 rate
    },
    # Sustained efficiencies to approximate scheduler, memory, and kernel overheads
    "efficiency": {
        "fp16": 0.35,
        "fp32": 0.35,
        "fp64": 0.2
    },
    # Fraction of the GPU we expect to have available for this job
    "allocation": 0.9
}

m4_pro_specs = {
    "name": "Apple M4 Pro",
    "frequency": 2.6e9,
    "flops_per_cycle": 24,  # rough FPC per core, fused-multiply-add
    "core_count": 10,  # 10 performance cores
    "efficiency": 0.65 
}
