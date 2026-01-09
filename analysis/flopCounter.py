import torch.nn as nn

# simple counting utilities for linear layers and activations

def count_linear_macs(layer: nn.Linear, batch_size: int = 1) -> int:
    """
    Count multiply-accumulate operations (MACs) for a Linear layer.

    MACs per sample = in_features * out_features
    MACs per batch  = batch_size * in_features * out_features
    """
    in_f = layer.in_features
    out_f = layer.out_features
    return batch_size * in_f * out_f


def count_linear_ops(
    layer: nn.Linear, batch_size: int = 1, include_bias: bool = True
) -> tuple[int, int]:
    """
    Count MACs and bias adds for a Linear layer.

    Returns:
        macs, bias_ops
    """
    macs = count_linear_macs(layer, batch_size=batch_size)
    # bias adds are optional and only counted if the layer has a bias term
    bias_ops = batch_size * layer.out_features if include_bias and layer.bias is not None else 0
    return macs, bias_ops


def count_activation_ops(num_elements: int, ops_per_element: int = 1) -> int:
    """
    Count elementwise activation operations.

    For ReLU/tanh/sigmoid we approximate as 1 op per element; adjust ops_per_element
    if you need a different cost model.
    """
    return num_elements * ops_per_element


def count_model_ops(
    model: nn.Module,
    batch_size: int = 1,
    include_bias: bool = True,
    include_activations: bool = False,
    activation_ops_per_element: int = 1,
) -> tuple[int, dict]:
    """
    Count total operations for all Linear layers in a model (forward pass).

    - MACs: batch_size * in_features * out_features per Linear
    - Bias adds: batch_size * out_features (optional)
    - Activations: assumes one activation after every Linear except the last,
      costing activation_ops_per_element per output element (optional).

    Returns:
        total_ops, breakdown dict with keys macs, bias_ops, activation_ops
    """
    macs = 0
    bias_ops = 0
    activation_ops = 0
    # record output sizes to estimate activation work later
    linear_output_sizes = []

    # walk all submodules and count only Linear layers
    for module in model.modules():
        if isinstance(module, nn.Linear):
            m, b = count_linear_ops(module, batch_size=batch_size, include_bias=include_bias)
            macs += m
            bias_ops += b
            linear_output_sizes.append(batch_size * module.out_features)

    if include_activations and linear_output_sizes:
        # assume activations after every linear except the final layer
        for elems in linear_output_sizes[:-1]:
            activation_ops += count_activation_ops(elems, ops_per_element=activation_ops_per_element)

    total_ops = macs + bias_ops + activation_ops
    breakdown = {
        "macs": macs,
        "bias_ops": bias_ops,
        "activation_ops": activation_ops,
    }
    return total_ops, breakdown


def count_mlp_macs(
    model: nn.Module,
    batch_size: int = 1,
    include_bias: bool = True,
    include_activations: bool = False,
    activation_ops_per_element: int = 1,
    return_breakdown: bool = False,
) -> int | dict:
    """
    Convenience wrapper for MLP-style models composed of Linear layers.

    Returns MACs by default. Set return_breakdown=True to receive a dict with
    macs, bias_ops, activation_ops, and total_ops.
    """
    total_ops, breakdown = count_model_ops(
        model,
        batch_size=batch_size,
        include_bias=include_bias,
        include_activations=include_activations,
        activation_ops_per_element=activation_ops_per_element,
    )
    # expose total_ops for callers that want a full breakdown
    breakdown["total_ops"] = total_ops
    if return_breakdown:
        return breakdown
    return breakdown["macs"]


def macs_to_flops(macs: int, fma_cost: int = 2) -> int:
    """
    Convert MAC count to FLOPs.

    Args:
        macs: number of MAC operations.
        fma_cost: how many FLOPs to count per MAC.
                  Use 2 for "mul + add" or 1 if you treat FMA as 1 FLOP.
    """
    return macs * fma_cost


def ops_breakdown_to_flops(breakdown: dict, fma_cost: int = 2) -> int:
    """
    Convert a breakdown dict (from count_mlp_macs(..., return_breakdown=True))
    to total FLOPs using the chosen FMA cost for MACs.
    """
    macs = breakdown.get("macs", 0)
    bias_ops = breakdown.get("bias_ops", 0)
    activation_ops = breakdown.get("activation_ops", 0)
    # count macs as fma_cost and add elementwise ops
    return macs * fma_cost + bias_ops + activation_ops
