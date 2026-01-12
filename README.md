# Neural Network Hardware Simulator (BETA)

This is a very early prototype of a personal project exploring how to estimate the computational cost
of neural networks in terms of MACs and FLOPs.

Current Features:
- Simple 3-layer MLP in PyTorch
- MAC/FLOP counter for linear layers (optional bias + activations)
- Theoretical runtime estimation from hardware profiles
- Measured forward-pass benchmarking (CPU/GPU) with summary stats
- Normalized output payloads for UI integration (estimate/measurement/comparison)
- Config-driven hardware specs in config/hardware_specs.json
- Streamlit UI prototype for hardware/runtime comparisons

Streamlit UI:
- Install dependencies: `pip install -r requirements.txt`
- Run: `streamlit run frontendUI.py`

Intended Features:
- Add more layer types (conv, attention, normalization, etc.)
- Add energy modeling (power, memory, efficiency curves)
- Expand beyond MAC/FLOP counts with memory traffic, arithmetic intensity, and overhead terms for better runtime predictions
- Add more hardware models (FPGA/TPU/Neuromorphic, mobile)
- Build a UI for browsing models, hardware, and tradeoffs
- Improve validation/schema tooling for config + outputs
