"""
GPU resource management and batch size optimization.

Automatically detects GPU capabilities and adjusts batch sizes
for each attribution method accordingly.
Includes thermal monitoring and throttling to prevent crashes.
"""

import torch
import time
import subprocess
from typing import Dict, Optional


class GPUManager:
    """Manages GPU resources and optimizes batch sizes."""

    def __init__(self):
        self.device = self._detect_device()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.batch_sizes = self._calculate_optimal_batches()
        self._last_temp_check = 0.0
        self._temp_check_interval = 20.0  # Check temperature every 20 seconds
        self._last_temp = None
        self._throttle_factor = 1.0  # Multiplier for batch sizes (1.0 = no throttling)

    # ------------------------------
    # Device / memory helpers
    # ------------------------------
    def _detect_device(self) -> str:
        """Detect optimal device: CUDA vs CPU."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def supports_fp16(self) -> bool:
        """
        Check if the current GPU architecture supports fast FP16 math.
        Volta (7.0) and above have native FP16; modern RTX GPUs are supported
        """
        if not torch.cuda.is_available():
            return False
        try:
            major, minor = torch.cuda.get_device_capability(0)
            return major >= 7
        except Exception:
            # Conservative default: assume supported on modern setups
            return True

    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9

    def _calculate_optimal_batches(self) -> Dict[str, int]:
        """Calculate optimal batch size for each attribution method."""
        # Optimized batch sizes for better VRAM utilization
        # RTX 5090 with 25.7GB can handle larger batches
        base_sizes = {
            "saliency": 32,
            "inputxgradient": 64,  # Increased for better VRAM usage
            "smoothgrad": 24,  # Increased for better VRAM usage
            "guided_backprop": 64,  # Increased for better VRAM usage
            "integrated_gradients": 16,  # Increased for better VRAM usage
            "gradientshap": 8,  # Increased for better VRAM usage
            "occlusion": 24,  # Increased for better VRAM usage
            "xrai": 8,  # Increased for better VRAM utilization - can handle batches now
            "grad_cam": 32,  # Increased for better VRAM usage
            "guided_gradcam": 32,  # Increased for better VRAM usage
            "random_baseline": 128,  # Increased for better VRAM usage
            "c3f": 2,  # Keep small - very memory intensive
        }

        # More aggressive scaling for high-VRAM GPUs (RTX 5090)
        if self.gpu_memory_gb >= 24:
            # RTX 5090 can handle 2x-2.5x base sizes safely
            return {k: int(v * 2.0) for k, v in base_sizes.items()}
        elif self.gpu_memory_gb > 16:
            return {k: int(v * 1.5) for k, v in base_sizes.items()}
        elif self.gpu_memory_gb > 8:
            return base_sizes
        else:
            return {k: max(1, v // 2) for k, v in base_sizes.items()}

    def get_batch_size(self, method: str) -> int:
        """Get optimal batch size for specific method."""
        return self.batch_sizes.get(method, 1)

    def get_strategy(self, method: str) -> str:
        """Get processing strategy for method."""
        if method in ["integrated_gradients", "gradientshap", "xrai"]:
            return "micro"  # XRAI now supports micro-batching
        elif method in ["c3f"]:
            return "single"
        else:
            return "batch"

    # ------------------------------
    # Backwards-compat utility methods
    # ------------------------------
    def get_memory_usage(self):
        """
        Return (total_gb, usage_percent) for CUDA device 0.
        Falls back to (0.0, 0.0) on CPU.
        """
        if not torch.cuda.is_available():
            return 0.0, 0.0
        total_bytes = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        usage = (allocated / total_bytes) * 100.0 if total_bytes > 0 else 0.0
        return total_bytes / 1e9, usage

    def clear_cache_if_needed(self, threshold_percent: float = 75.0) -> None:
        """
        If current CUDA memory usage exceeds threshold, clear CUDA cache.
        No-op on CPU.
        """
        if not torch.cuda.is_available():
            return
        _, usage = self.get_memory_usage()
        if usage >= threshold_percent:
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def get_optimal_inference_batch_size(self) -> int:
        """
        Provide a coarse default inference batch size based on GPU memory.
        Optimized for better VRAM utilization on high-memory GPUs.
        Aggressively increased for GPUs with 23.5GB+ VRAM to maximize utilization.
        """
        # Apply throttle factor for thermal management
        base_size = 64
        if self.gpu_memory_gb >= 23:
            # RTX 3050/5090 with 23.5GB+ can handle much larger batches
            # With 0% GPU utilization, we can be very aggressive
            base_size = 768  # Significantly increased for better VRAM utilization
        elif self.gpu_memory_gb >= 16:
            base_size = 512  # Increased for high-VRAM GPUs
        elif self.gpu_memory_gb > 8:
            base_size = 256  # Increased for mid-range GPUs
        else:
            base_size = 64
        
        return int(base_size * self._throttle_factor)

    def get_safe_batch_size(self, desired: int, current_usage_percent: float) -> int:
        """
        Adjust desired batch size based on current usage. Simple heuristic.
        """
        if current_usage_percent < 85.0:
            return max(1, desired)
        if current_usage_percent < 92.0:
            return max(1, desired // 2)
        return max(1, desired // 4)

    def get_gpu_temperature(self) -> Optional[float]:
        """Get current GPU temperature in Celsius using nvidia-smi."""
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
        Should be called periodically during long-running operations.
        """
        current_time = time.time()
        if current_time - self._last_temp_check < self._temp_check_interval:
            return
        
        self._last_temp_check = current_time
        temp = self.get_gpu_temperature()
        
        if temp is None:
            return  # Can't monitor temperature
        
        self._last_temp = temp
        
        # Thermal throttling thresholds
        if temp >= 87:
            # Critical: reduce workload significantly
            self._throttle_factor = 0.3
            import logging
            logging.warning(f"GPU temperature critical: {temp}°C - throttling to 30% capacity")
        elif temp >= 83:
            # High: reduce workload moderately
            self._throttle_factor = 0.5
            import logging
            logging.warning(f"GPU temperature high: {temp}°C - throttling to 50% capacity")
        elif temp >= 78:
            # Moderate: slight reduction
            self._throttle_factor = 0.7
        elif temp < 75:
            # Normal: restore full capacity gradually
            if self._throttle_factor < 1.0:
                self._throttle_factor = min(1.0, self._throttle_factor + 0.1)

    def sync_and_clear(self) -> None:
        """Synchronize GPU operations and clear cache. Helps prevent crashes."""
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def print_info(self):
        """Print GPU information."""
        import logging
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
