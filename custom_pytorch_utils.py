import torch

def unfold_custom(input_tensor, dimension, size, step):
    if dimension < 0:
        dimension += input_tensor.dim()

    assert 0 <= dimension < input_tensor.dim(), "Invalid dimension"

    input_size = input_tensor.size(dimension)
    shape = list(input_tensor.shape)

    num_patches = torch.div(input_size - size + step, step, rounding_mode='trunc')
    new_shape = shape[:dimension] + [num_patches, size] + shape[dimension+1:]

    unfolded_tensor = torch.empty(new_shape, dtype=input_tensor.dtype)

    for i in range(num_patches):
        start = i * step
        end = start + size
        if dimension == 0:
            unfolded_tensor[i] = input_tensor[start:end]
        else:
            unfolded_tensor.select(dimension, i).copy_(input_tensor.narrow(dimension, start, size))

    return unfolded_tensor