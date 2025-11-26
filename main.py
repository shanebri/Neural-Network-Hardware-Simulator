import torch
from models.mlp import SimpleMLP
from analysis.flopCounter import count_mlp_macs, macs_to_flops
import time
from analysis.hardware_estimates import compute_flop_rate_cpu, compute_flop_rate_gpu, estimate_runtime
from models.hardware_specs import cpu_specs, gpu_specs, m4_pro_specs


#define model and dummy input for model
model = SimpleMLP()

#this represents how many input samples the model processes in parallel
batch_size = 32

#shape: [batch_size, 28x28]
#each row can be considered a flattened grayscale 28x28 image
x = torch.randn(batch_size, 784)

# Time the forward pass
start = time.perf_counter()

#disabed gradient tracking
#measuring inference time not training so gradients and backpropagation is not needed
with torch.no_grad():
    y = model(x)
end = time.perf_counter()

#y shape should be [batch_size, 10] for 10 output classes
print("Output shape:", y.shape)
print("Forward pass time: (cpu)", end - start)

#count THEORETICAL MACs / FLOPs
macs = count_mlp_macs(model, batch_size)
flops = macs_to_flops(macs)

print("MACs:", macs)
print("FLOPs:", flops)
print("FLOPs per sample:", flops / batch_size)

#these FLOPs are "workload" estimates that
#will be combined with assumed hardware capabilities
#to estimate execution time and energy of different architectures



cpu_rate = compute_flop_rate_cpu(cpu_specs)
gpu_rate = compute_flop_rate_gpu(gpu_specs)
apple_m4_rate = compute_flop_rate_cpu(m4_pro_specs)

print("CPU estimate: (Intel i9 9900k)", estimate_runtime(flops, cpu_rate))
print("GPU estimate: (NVIDIA GeForce RTX 5070Ti)", estimate_runtime(flops, gpu_rate))
print("MacBook estimate: (Apple M4 Pro Chip)", estimate_runtime(flops, apple_m4_rate))