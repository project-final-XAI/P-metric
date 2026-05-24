"""
GPU utility functions for clear separation of GPU operations from business logic.
Optimized for accurate VRAM accounting and asynchronous data transfers.
"""

import torch
from typing import List, Optional, Tuple, Union

# Constant for binary Gigabyte conversion (GiB)
_GIB = 1024 ** 3


def get_memory_usage() -> Tuple[float, float, float]:
    """
    Get current GPU memory metrics.

    Returns:
        Tuple of (total_gib, allocated_percent, reserved_percent).
        Returns (0.0, 0.0, 0.0) on CPU.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    device_id = torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(device_id).total_memory
    allocated_bytes = torch.cuda.memory_allocated(device_id)
    reserved_bytes = torch.cuda.memory_reserved(device_id)

    if total_bytes == 0:
        return 0.0, 0.0, 0.0

    allocated_pct = (allocated_bytes / total_bytes) * 100.0
    reserved_pct = (reserved_bytes / total_bytes) * 100.0

    return total_bytes / _GIB, allocated_pct, reserved_pct


def clear_cache_if_needed(threshold_percent: float = 75.0) -> None:
    """
    Clear CUDA cache if memory reservation or allocation exceeds threshold.
    """
    if not torch.cuda.is_available():
        return

    _, allocated_pct, reserved_pct = get_memory_usage()
    # Check both to see if PyTorch's internal cache block footprint is getting tight
    if max(allocated_pct, reserved_pct) >= threshold_percent:
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def sync_and_clear() -> None:
    """Synchronize GPU operations and clear cache securely."""
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def prepare_batch_tensor(
    images: Union[List[torch.Tensor], torch.Tensor],
    device: str,
    use_fp16: bool = False,
    memory_format: Optional[torch.memory_format] = torch.channels_last
) -> torch.Tensor:
    """
    Stack images into batch tensor with advanced pipeline optimizations.
    """
    is_cuda = (device == "cuda" or "cuda" in str(device))

    if isinstance(images, torch.Tensor):
        batch_tensor = images.unsqueeze(0) if images.ndim == 3 else images
    else:
        # Fast assumption: check the first item's device instead of executing a python loop over all
        if is_cuda and images[0].device.type == "cuda":
            batch_tensor = torch.stack(images)
        else:
            # If coming from CPU, pin the stacked memory to accelerate the subsequent .to(non_blocking=True)
            batch_tensor = torch.stack(images)
            if is_cuda and not batch_tensor.is_pinned():
                try:
                    batch_tensor = batch_tensor.pin_memory()
                except Exception:
                    pass

    # Move to target device asynchronously
    if batch_tensor.device.type != "cuda" and is_cuda:
        batch_tensor = batch_tensor.to(device, non_blocking=True)
    elif batch_tensor.device.type != device:
        batch_tensor = batch_tensor.to(device)

    # Convert memory layout (channels_last maximizes Tensor Core performance on modern GPUs)
    if memory_format is not None and batch_tensor.ndim == 4:
        batch_tensor = batch_tensor.to(memory_format=memory_format)

    # Precision casting
    if use_fp16 and is_cuda:
        batch_tensor = batch_tensor.half()

    return batch_tensor