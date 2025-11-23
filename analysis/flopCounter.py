def count_linear_macs(layer, batch_size=1):
    """
    counts the number of multipy-accumulate operations (MACs)
    for a single layer for a given batch size

    for a linear layer with:
        in_features = N_in
        out_features = N_out

    each neuron does a dot product with length N_in:
        N_in multiplications
        N_in - 1 additions
    this is approximately counted as N_in MACs per output neuron

    each of the N_out output neurons does N_in MACs
    so MACs per sample layer = N_in * N_out

    the MACs per batch scale linearly so:
    MACs per batch = B * N_in * N_out
    """
    in_f = layer.in_features
    out_f = layer.out_features
    return batch_size * in_f * out_f

def count_mlp_macs(model, batch_size=1):

    #total MACs is just all MACs per linear layer added together
    macs = 0
    macs += count_linear_macs(model.fc1, batch_size)
    macs += count_linear_macs(model.fc2, batch_size)
    macs += count_linear_macs(model.fc3, batch_size)
    return macs

def macs_to_flops(macs): #each flop is 2 macs
    """
    1 MAC ≈ 1 multiply + 1 add which is 2 floating point operations so:
    1 FLOP is just 2 MACs
    """
    return 2 * macs