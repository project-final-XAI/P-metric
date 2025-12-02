"""
GPU resource management and batch size optimization.

Automatically detects GPU capabilities and adjusts batch sizes for each attribution method.
Includes thermal monitoring and throttling to prevent crashes.
"""

import torch
import time
import subprocess
import logging
from typing import Dict, Optional

from core.gpu_utils import get_memory_usage, clear_cache_if_needed, sync_and_clear


class GPUManager:
    """
    Manages GPU resources and optimizes batch sizes.
    
    Provides methods for:
    - GPU capability detection
    - Batch size calculation based on GPU memory
    - Thermal monitoring and throttling
    - Memory management
    """
    
    def __init__(self):
        """Initialize GPU manager with device detection and batch size calculation."""
        self.device = self._detect_device()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.batch_sizes = self._calculate_optimal_batches()
        
        # Thermal monitoring state
        self._last_temp_check = 0.0
        self._temp_check_interval = 40.0  # Check temperature every X seconds
        self._last_temp = None
        self._throttle_factor = 1.2  # Multiplier for batch sizes (1.0 = no throttling)
    
    # ------------------------------ Device Detection ------------------------------
    
    def _detect_device(self) -> str:
        """
        Detect optimal device: CUDA vs CPU.
        
        Returns:
            'cuda' if CUDA is available, 'cpu' otherwise
        """
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    def _get_gpu_memory(self) -> float:
        """
        Get available GPU memory in GB.
        
        Returns:
            GPU memory in GB, or 0.0 if CUDA is not available
        """
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    
    def supports_fp16(self) -> bool:
        """
        Check if the current GPU architecture supports fast FP16 math.
        
        Volta (compute capability 7.0) and above have native FP16 support.
        Modern RTX GPUs (Turing, Ampere, Ada) are supported.
        
        Returns:
            True if GPU supports FP16, False otherwise
        """
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            return major >= 7
        except Exception:
            # Conservative default: assume supported on modern setups
            return True
    
    # ------------------------------ Batch Size Calculation ------------------------------
    
    def _calculate_optimal_batches(self) -> Dict[str, int]:
        """
        Calculate optimal batch size for each attribution method.
        
        Batch sizes are scaled based on GPU memory:
        - High-VRAM GPUs (24GB+): 2x base sizes
        - Mid-VRAM GPUs (16-24GB): 1.5x base sizes
        - Standard GPUs (8-16GB): base sizes
        - Low-VRAM GPUs (<8GB): 0.5x base sizes
        
        Returns:
            Dictionary mapping method names to batch sizes
        """
        # Base batch sizes for each attribution method
        # These are conservative defaults that work on most GPUs
        base_sizes = {
            "saliency": 32,
            "inputxgradient": 64,
            "smoothgrad": 24,
            "guided_backprop": 64,
            "integrated_gradients": 16,
            "gradientshap": 8,
            "occlusion": 24,
            "xrai": 8,
            "grad_cam": 32,
            "guided_gradcam": 32,
            "random_baseline": 128,
            "c3f": 1,  # Must be 1 - C3F processes one image at a time
        }
        
        # Scale based on GPU memory, but keep batch_size=1 methods unchanged
        # Methods like C3F require batch_size=1 and should not be scaled
        if self.gpu_memory_gb >= 24:
            # High-VRAM GPUs can handle 2x base sizes
            return {k: (v if v == 1 else int(v * 2.0)) for k, v in base_sizes.items()}
        elif self.gpu_memory_gb > 16:
            # Mid-VRAM GPUs can handle 1.5x base sizes
            return {k: (v if v == 1 else int(v * 1.5)) for k, v in base_sizes.items()}
        elif self.gpu_memory_gb > 8:
            # Standard GPUs use base sizes
            return base_sizes
        else:
            # Low-VRAM GPUs use half base sizes
            return {k: max(1, v // 2) for k, v in base_sizes.items()}
    
    def get_batch_size(self, method: str) -> int:
        """
        Get optimal batch size for specific attribution method.
        
        Args:
            method: Name of the attribution method
            
        Returns:
            Optimal batch size for the method
        """
        return self.batch_sizes.get(method, 1)
    
    # ------------------------------ Memory Management ------------------------------
    
    # Lookup table for memory-based batch size multipliers (cleaner than nested ifs)
    _MEMORY_USAGE_MULTIPLIERS = [
        (20.0, 4.0),   # < 20% usage
        (35.0, 3.0),   # < 35% usage
        (50.0, 2.5),   # < 50% usage
        (70.0, 2.0),   # < 70% usage
        (85.0, 1.0),   # < 85% usage
        (92.0, 0.5),   # < 92% usage
        (100.0, 0.25), # >= 92% usage
    ]
    
    def get_optimal_inference_batch_size(self, current_memory_usage: float = None) -> int:
        """
        Get optimal inference batch size based on GPU memory.
        
        This is used for Phase 2 evaluation where we process many occluded images.
        Batch sizes are aggressively increased for high-VRAM GPUs to maximize utilization.
        
        Returns:
            Optimal batch size (adjusted for all factors)
        """
        # Base batch sizes by total GPU memory
        if self.gpu_memory_gb >= 22:
            base_size = 512   # Very high-VRAM GPUs
        elif self.gpu_memory_gb >= 16:
            base_size = 384   # High-VRAM GPUs
        elif self.gpu_memory_gb > 8:
            base_size = 256   # Mid-VRAM GPUs
        else:
            base_size = 64    # Low-VRAM GPUs
        
        # Apply thermal throttling first
        base_size = int(base_size * self._throttle_factor)
        
        # Apply memory pressure adjustment (using lookup table)
        if current_memory_usage is None:
            _, current_memory_usage = self.get_memory_usage()
        
        # If VRAM is full (95%+), reduce significantly  
        if current_memory_usage >= 95.0:
            memory_multiplier = 0.1  # Reduce to 10% of the original size
        else:
            memory_multiplier = 1.0
            for threshold, multiplier in self._MEMORY_USAGE_MULTIPLIERS:
                if current_memory_usage < threshold:
                    memory_multiplier = multiplier
                    break
        
        # Calculate final batch size with cap
        final_size = int(base_size * memory_multiplier)
        max_cap = 2048  # Hard cap to avoid excessive batches
        
        return max(1, min(final_size, max_cap))
    
    def get_safe_batch_size(self, desired: int, current_usage_percent: float) -> int:
        """
        Adjust desired batch size based on current GPU memory usage.
        
        Args:
            desired: Desired batch size
            current_usage_percent: Current GPU memory usage percentage (0-100)
            
        Returns:
            Safe batch size (at least 1)
        """
        if current_usage_percent < 85.0:
            return max(1, desired)
        elif current_usage_percent < 92.0:
            return max(1, desired // 2)
        else:
            return max(1, desired // 4)
    
    # ------------------------------ Thermal Management ------------------------------
    
    def get_gpu_temperature(self) -> Optional[float]:
        """
        Get current GPU temperature in Celsius using nvidia-smi.
        
        Returns:
            GPU temperature in Celsius, or None if unavailable
        """
        if not torch.cuda.is_available():
            return None
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                temp_str = result.stdout.strip()
                if temp_str:
                    return float(temp_str)
        except Exception:
            pass
        return None
    
    def check_and_throttle(self) -> None:
        """
        Check GPU temperature and adjust throttle factor if needed.
        
        Thermal throttling thresholds:
        - >= 87°C: Critical - throttle to 30% capacity
        - >= 83°C: High - throttle to 50% capacity
        - >= 78°C: Moderate - throttle to 70% capacity
        - < 75°C: Normal - gradually restore full capacity
        
        Should be called periodically during long-running operations.
        """
        current_time = time.time()
        
        # Only check temperature every N seconds to avoid overhead
        if current_time - self._last_temp_check < self._temp_check_interval:
            return
        
        self._last_temp_check = current_time
        temp = self.get_gpu_temperature()
        
        if temp is None:
            return  # Can't monitor temperature
        
        self._last_temp = temp
        
        # Apply thermal throttling based on temperature
        if temp >= 87:
            # Critical: reduce workload significantly
            self._throttle_factor = 0.3
            logging.warning(f"GPU temperature critical: {temp}°C - throttling to 30% capacity")
        elif temp >= 83:
            # High: reduce workload moderately
            self._throttle_factor = 0.5
            logging.warning(f"GPU temperature high: {temp}°C - throttling to 50% capacity")
        elif temp >= 78:
            # Moderate: slight reduction
            self._throttle_factor = 0.7
        elif temp < 75:
            # Normal: restore full capacity gradually
            if self._throttle_factor < 1.0:
                self._throttle_factor = min(1.0, self._throttle_factor + 0.1)
    
    # ------------------------------ Information ------------------------------
    
    def print_info(self):
        """Print GPU information for debugging."""
        logging.info(f"Device: {self.device}")
        if self.device == "cuda":
            logging.info(f"GPU Memory: {self.gpu_memory_gb:.1f}GB")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            temp = self.get_gpu_temperature()
            if temp is not None:
                logging.info(f"GPU Temperature: {temp}°C")
        else:
            logging.info("Running on CPU - this will be slower")
        logging.info("=" * 60)
