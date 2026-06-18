# https://github.com/BobJohnson24/ComfyUI-INT8-Fast/blob/main/convrot.py
# Group-wise Hadamard rotation for improved INT8 quantization quality.
# Reduces outliers in activations/weights before quantization.

import torch

_HADAMARD_CACHE: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}


def build_hadamard(size: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Build a (size x size) Hadamard matrix via Sylvester construction.
    `size` must be a power of 2.
    """
    key = (size, device, dtype)
    if key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[key]

    assert size > 0 and (size & (size - 1)) == 0, "size must be a power of 2"

    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < size:
        H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)

    H = H / (size ** 0.5)
    _HADAMARD_CACHE[key] = H
    return H


def rotate_weight(weight: torch.Tensor, H: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Apply group-wise Hadamard rotation to a weight tensor (in-place on a copy).
    weight: [out_features, in_features]
    H:      [group_size, group_size]
    Returns a float32 rotated weight.
    """
    w = weight.float()
    in_features = w.shape[1]

    if in_features % group_size != 0:
        return w  # Cannot rotate, return as-is

    num_groups = in_features // group_size
    # Reshape to [..., num_groups, group_size], apply H, reshape back
    w = w.reshape(w.shape[0], num_groups, group_size)
    w = torch.einsum("omg,gh->omh", w, H)
    return w.reshape(weight.shape[0], in_features)


def rotate_activation(x: torch.Tensor, H: torch.Tensor, group_size: int) -> torch.Tensor:
    """
    Apply group-wise Hadamard rotation to an activation tensor.
    x: [..., in_features]
    H: [group_size, group_size]
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]

    if in_features % group_size != 0:
        return x  # Cannot rotate, return as-is

    num_groups = in_features // group_size
    x_f = x.float().reshape(-1, num_groups, group_size)
    x_f = torch.einsum("bmg,gh->bmh", x_f, H)
    return x_f.reshape(orig_shape).to(x.dtype)
