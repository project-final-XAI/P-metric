"""
GPU resource management and batch size optimization.
Utilizes low-overhead native NVML bindings for thermal tracking.
"""

import torch
import time
import subprocess
import logging
from typing import Dict, Optional

from core.gpu_utils import get_memory_usage

# Try using high-performance NVML bindings over slow subprocesses
try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False

try:
    import config as _cfg
except ImportError:
    _cfg = None

_VRAM_HIGH = getattr(_cfg, 'VRAM_TIER_HIGH', 22.0)
_VRAM_MID = getattr(_cfg, 'VRAM_TIER_MID', 16.0)
_VRAM_LOW = getattr(_cfg, 'VRAM_TIER_LOW', 8.0)


class GPUManager:
    """Manages GPU resources, adjusts throughput profiles, and handles thermal monitoring."""

    _MEMORY_USAGE_MULTIPLIERS = [
        (20.0, 4.0), (35.0, 3.0), (50.0, 2.5),
        (70.0, 2.0), (85.0, 1.0), (92.0, 0.5), (100.0, 0.25)
    ]

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_memory_gb = self._get_gpu_memory()
        self.batch_sizes = self._calculate_optimal_batches()

        self._last_temp_check = 0.0
        self._temp_check_interval = 5.0
        self._last_temp = None
        self._throttle_factor = 1.0  # FIXED: Corrected initialization from 1.2 to 1.0

    def _get_gpu_memory(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    def supports_fp16(self) -> bool:
        if not torch.cuda.is_available():
            return False
        try:
            return torch.cuda.get_device_capability(0)[0] >= 7
        except Exception:
            return True

    def _calculate_optimal_batches(self) -> Dict[str, int]:
        base_sizes = {
            "saliency": 64, "inputxgradient": 64, "smoothgrad": 8,
            "guided_backprop": 64, "integrated_gradients": 16, "gradientshap": 8,
            "occlusion": 24, "xrai": 4, "grad_cam": 32, "guided_gradcam": 32,
            "random_baseline": 128,
        }

        if self.gpu_memory_gb >= _VRAM_HIGH:
            multiplier = 2.0
        elif self.gpu_memory_gb > _VRAM_MID:
            multiplier = 1.5
        elif self.gpu_memory_gb > _VRAM_LOW:
            return base_sizes
        else:
            return {k: max(1, v // 2) for k, v in base_sizes.items()}

        return {k: (v if v == 1 else int(v * multiplier)) for k, v in base_sizes.items()}

    def get_batch_size(self, method: str) -> int:
        return self.batch_sizes.get(method, 1)

    def get_optimal_inference_batch_size(self, current_memory_usage: float = None) -> int:
        if self.gpu_memory_gb >= _VRAM_HIGH:
            base_size = 512
        elif self.gpu_memory_gb >= _VRAM_MID:
            base_size = 384
        elif self.gpu_memory_gb > _VRAM_LOW:
            base_size = 256
        else:
            base_size = 64

        # Apply current thermal limitations
        base_size = int(base_size * self._throttle_factor)

        if current_memory_usage is None:
            _, allocated_pct, reserved_pct = get_memory_usage()
            current_memory_usage = max(allocated_pct, reserved_pct)

        if current_memory_usage >= 95.0:
            memory_multiplier = 0.1
        else:
            memory_multiplier = 1.0
            for threshold, multiplier in self._MEMORY_USAGE_MULTIPLIERS:
                if current_memory_usage < threshold:
                    memory_multiplier = multiplier
                    break

        return max(1, min(int(base_size * memory_multiplier), 2048))

    def get_safe_batch_size(self, desired: int, current_usage_percent: float) -> int:
        if current_usage_percent < 85.0:
            return max(1, desired)
        return max(1, desired // 2) if current_usage_percent < 92.0 else max(1, desired // 4)

    def get_gpu_temperature(self) -> Optional[float]:
        """Reads GPU temperature using fast native NVML library with subprocess fallback."""
        if not torch.cuda.is_available():
            return None

        if _NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                return float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            except Exception:
                pass

        # Subprocess Fallback Strategy
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass
        return None

    def check_and_throttle(self) -> None:
        current_time = time.time()
        if current_time - self._last_temp_check < self._temp_check_interval:
            return

        self._last_temp_check = current_time
        temp = self.get_gpu_temperature()

        if temp is None:
            return

        self._last_temp = temp

        if temp >= 87:
            self._throttle_factor = 0.3
            logging.warning(f"GPU temperature critical: {temp}°C - throttling to 30% capacity")
        elif temp >= 83:
            self._throttle_factor = 0.5
            logging.warning(f"GPU temperature high: {temp}°C - throttling to 50% capacity")
        elif temp >= 78:
            self._throttle_factor = 0.7
        elif temp < 75 and self._throttle_factor < 1.0:
            self._throttle_factor = min(1.0, self._throttle_factor + 0.1)

    def print_info(self):
        logging.info(f"Device: {self.device}")
        if self.device == "cuda":
            logging.info(f"GPU Memory: {self.gpu_memory_gb:.1f} GiB")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
            temp = self.get_gpu_temperature()
            if temp is not None:
                logging.info(f"GPU Temperature: {temp}°C")
        else:
            logging.info("Running on CPU")
        logging.info("=" * 60)