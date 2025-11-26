def compute_flop_rate_cpu(specs):
    return specs["frequency"] * specs["flops_per_cycle"]

def compute_flop_rate_gpu(specs):
    return specs["frequency"] * specs["num_cores"] * specs["flops_per_core"]

def estimate_runtime(total_ops, flop_rate):
    return total_ops/flop_rate
