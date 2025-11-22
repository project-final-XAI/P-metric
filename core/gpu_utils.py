"""
GPU utility functions for clear separation of GPU operations from business logic.

This module provides helper functions for common GPU operations like memory management,
tensor transfers, batch size calculations, and thermal monitoring. All GPU-specific
code is centralized here for better readability and maintainability.
"""

import torch
import logging
from typing import List, Tuple, Optional


def get_memory_usage() -> Tuple[float, float]:
    """
    Get current GPU memory usage.
    
    Returns:
        Tuple of (total_gb, usage_percent). Returns (0.0, 0.0) on CPU.
    """
    if not torch.cuda.is_available():
        return 0.0, 0.0
    
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    usage_percent = (allocated / total_bytes) * 100.0 if total_bytes > 0 else 0.0
    return total_bytes / 1e9, usage_percent


def clear_cache_if_needed(threshold_percent: float = 75.0) -> None:
    """
    Clear CUDA cache if memory usage exceeds threshold.
    
    This helps prevent out-of-memory errors by freeing unused GPU memory.
    Only clears cache if usage is above threshold to avoid unnecessary overhead.
    
    Args:
        threshold_percent: Memory usage threshold (0-100) to trigger cache clear
    """
    if not torch.cuda.is_available():
        return
    
    _, usage = get_memory_usage()
    if usage >= threshold_percent:
        torch.cuda.empty_cache()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def sync_and_clear() -> None:
    """
    Synchronize GPU operations and clear cache.
    
    This ensures all GPU operations complete before clearing cache,
    which helps prevent crashes during long-running operations.
    """
    if not torch.cuda.is_available():
        return
    
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def transfer_to_device(
    tensor: torch.Tensor,
    device: str,
    non_blocking: bool = True,
    memory_format: Optional[torch.memory_format] = None
) -> torch.Tensor:
    """
    Transfer tensor to device with optional memory format optimization.
    
    Args:
        tensor: Input tensor
        device: Target device ('cuda' or 'cpu')
        non_blocking: Whether to use non-blocking transfer (faster, async)
        memory_format: Optional memory format (e.g., torch.channels_last for CNNs)
        
    Returns:
        Tensor on target device
    """
    result = tensor.to(device, non_blocking=non_blocking)
    
    if memory_format is not None and result.ndim == 4:
        try:
            result = result.to(memory_format=memory_format, non_blocking=non_blocking)
        except Exception:
            # Some tensors don't support memory format conversion
            pass
    
    return result


def prepare_batch_tensor(
    images: List[torch.Tensor],
    device: str,
    use_fp16: bool = False,
    memory_format: Optional[torch.memory_format] = None
) -> torch.Tensor:
    """
    Stack images into batch tensor with GPU optimizations.
    
    This function efficiently stacks images and applies GPU optimizations:
    - Non-blocking transfers for better pipelining
    - Channels-last memory format for CNNs (better TensorCore utilization)
    - FP16 conversion if supported
    
    Args:
        images: List of image tensors (each C, H, W)
        device: Target device
        use_fp16: Whether to convert to FP16 (faster inference on modern GPUs)
        memory_format: Optional memory format optimization
        
    Returns:
        Batch tensor of shape (B, C, H, W)
    """
    if device == "cuda":
        # Check if images are already on GPU - stack directly if so
        if all(img.device.type == "cuda" for img in images):
            batch_tensor = torch.stack(images)
        else:
            # Transfer with non-blocking for better pipelining
            batch_tensor = torch.stack(images).to(device, non_blocking=True)
    else:
        batch_tensor = torch.stack(images).to(device)
    
    # Apply memory format optimization (channels_last improves CNN performance)
    if memory_format is not None and batch_tensor.ndim == 4:
        try:
            batch_tensor = batch_tensor.to(memory_format=memory_format, non_blocking=True)
        except Exception:
            pass
    
    # Convert to FP16 if requested (faster inference, less memory)
    if use_fp16 and device == "cuda":
        batch_tensor = batch_tensor.half()
    
    return batch_tensor


def calculate_safe_batch_size(
    desired_size: int,
    current_memory_usage_percent: float,
    throttle_factor: float = 1.0
) -> int:
    """
    Calculate safe batch size based on current GPU memory usage.
    
    This function adjusts batch size dynamically based on:
    - Current memory usage (reduce if high)
    - Thermal throttling factor (reduce if GPU is hot)
    
    Args:
        desired_size: Desired batch size
        current_memory_usage_percent: Current GPU memory usage (0-100)
        throttle_factor: Thermal throttling multiplier (0-1, 1.0 = no throttling)
        
    Returns:
        Safe batch size (at least 1)
    """
    # Apply thermal throttling first
    safe_size = int(desired_size * throttle_factor)
    
    # Adjust based on memory usage
    if current_memory_usage_percent < 85.0:
        return max(1, safe_size)
    elif current_memory_usage_percent < 92.0:
        return max(1, safe_size // 2)
    else:
        return max(1, safe_size // 4)


def warmup_gpu(device: str, sample_shape: Tuple[int, ...], use_fp16: bool = False) -> None:
    """
    Warm up GPU memory allocator to reduce first-batch allocation overhead.
    
    This pre-allocates a small tensor to initialize the CUDA context and
    memory allocator, reducing latency for the first real batch.
    
    Args:
        device: Target device
        sample_shape: Shape of sample tensor (without batch dimension)
        use_fp16: Whether to use FP16 (should match inference dtype)
    """
    if device != "cuda":
        return
    
    try:
        dtype = torch.float16 if use_fp16 else torch.float32
        _ = torch.zeros((1, *sample_shape), device=device, dtype=dtype)
        del _
        torch.cuda.empty_cache()
    except Exception:
        # If warmup fails, continue normally
        pass
