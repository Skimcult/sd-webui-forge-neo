import torch

from backend import memory_management
from modules.devices import device


def stream_context(stream_device=None):
    dev = stream_device or device
    if memory_management.is_device_cuda(dev):
        return torch.cuda.stream
    if memory_management.is_device_xpu(dev) and hasattr(torch, "xpu"):
        return torch.xpu.stream
    return None


def create_stream(stream_device=None, *, priority=0):
    dev = stream_device or device
    if memory_management.is_device_cuda(dev):
        return torch.cuda.Stream(device=dev, priority=priority)
    if memory_management.is_device_xpu(dev) and hasattr(torch, "xpu"):
        return torch.xpu.Stream(device=dev, priority=priority)
    return None


def get_current_stream():
    return memory_management.current_stream(device)


def get_new_stream():
    return memory_management.get_offload_stream(device)


def get_vae_stream():
    if torch.cuda.is_available():
        return torch.cuda.Stream(device=device, priority=1)
    if torch.xpu.is_available():
        return torch.xpu.Stream(device=device, priority=1)


def should_use_stream():
    return current_stream is not None and mover_stream is not None


current_stream = get_current_stream()
mover_stream = get_new_stream()
