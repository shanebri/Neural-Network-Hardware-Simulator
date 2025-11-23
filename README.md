# Neural Network Hardware Simulator (Alpha)

This is a very early prototype of a personal project exploring how to estimate the computational cost
of neural networks in terms of MACs and FLOPs.

Current Features:
- Simple 3-layer MLP in PyTorch
- MAC and FLOP counter for linear layers
- Script to time a forward pass and report workload

Intended Features:
- Add different hardware models other than CPU (GPU/FPGA/Neuromorphic Architecture)
- Estimate runtime and energy based on FLOP rate and power
- Add visualization of performance tradeoffs (maybe) 