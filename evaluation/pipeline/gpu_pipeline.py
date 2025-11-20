"""
GPU pipeline management for occlusion evaluation.

Handles GPU pre-allocation, memory management, and batch size optimization.
"""

import torch
import logging
from typing import Optional


class GPUPreparer:
    """
    Handles GPU pre-allocation and warm-up to reduce allocation overhead.
    """
    
    def __init__(self, device: str, use_fp16: bool = False):
        """
        Initialize GPU preparer.
        
        Args:
            device: Device to use ("cuda" or "cpu")
            use_fp16: Whether FP16 is enabled
        """
        self.device = device
        self.use_fp16 = use_fp16
    
    def warm_up(self, sample_image_shape: tuple) -> None:
        """
        Warm up GPU by pre-allocating a small tensor.
        
        This reduces first-batch allocation overhead during processing.
        
        Args:
            sample_image_shape: Shape of a sample image (without batch dimension)
        """
        if self.device != "cuda":
            return
        
        try:
            # Pre-allocate a small workspace tensor to warm up GPU memory allocator
            dtype = torch.float16 if self.use_fp16 else torch.float32
            _ = torch.zeros((1, *sample_image_shape), device=self.device, dtype=dtype)
            del _
            # Trigger a small dummy operation to initialize CUDA context
            torch.cuda.empty_cache()
        except Exception as e:
            logging.debug(f"GPU warm-up failed (non-critical): {e}")


class BatchSizeManager:
    """
    Manages batch size calculation based on GPU memory and thermal conditions.
    """
    
    def __init__(self, gpu_manager, config):
        """
        Initialize batch size manager.
        
        Args:
            gpu_manager: GPUManager instance
            config: Configuration object with INFERENCE_MAX_BATCH attribute
        """
        self.gpu_manager = gpu_manager
        self.config = config
    
    def get_optimal_batch_size(self) -> int:
        """
        Calculate optimal batch size based on current GPU conditions.
        
        Returns:
            Optimal batch size for inference
        """
        # Get base batch size from GPU manager
        base_size = self.gpu_manager.get_optimal_inference_batch_size()
        
        # Check memory usage
        _, current_usage = self.gpu_manager.get_memory_usage()
        
        # Adjust based on memory usage
        base_size = self.gpu_manager.get_safe_batch_size(base_size, current_usage)
        
        # Check GPU temperature and throttle if needed
        self.gpu_manager.check_and_throttle()
        throttle_factor = getattr(self.gpu_manager, '_throttle_factor', 1.0)
        base_size = int(base_size * throttle_factor)
        
        # Aggressive multipliers for better VRAM utilization when GPU is underutilized
        max_cap = getattr(self.config, "INFERENCE_MAX_BATCH", 2048)
        
        if current_usage < 20.0:
            # Very low usage - be very aggressive
            base_size = min(int(base_size * 4.0), max_cap)
        elif current_usage < 35.0:
            base_size = min(int(base_size * 3.0), max_cap)
        elif current_usage < 50.0:
            base_size = min(int(base_size * 2.5), max_cap)
        elif current_usage < 70.0:
            base_size = min(int(base_size * 2.0), max_cap)
        
        return base_size
    
    def should_clear_cache(self, threshold_percent: float = 80.0) -> bool:
        """
        Check if GPU cache should be cleared based on memory usage.
        
        Args:
            threshold_percent: Memory usage threshold percentage
        
        Returns:
            True if cache should be cleared
        """
        _, current_usage = self.gpu_manager.get_memory_usage()
        return current_usage >= threshold_percent
    
    def clear_cache_if_needed(self, threshold_percent: float = 80.0) -> None:
        """
        Clear GPU cache if memory usage exceeds threshold.
        
        Args:
            threshold_percent: Memory usage threshold percentage
        """
        if self.should_clear_cache(threshold_percent):
            self.gpu_manager.clear_cache_if_needed(threshold_percent=threshold_percent)

