"""
GPU resource management and batch size optimization.

Automatically detects GPU capabilities and adjusts batch sizes
for each attribution method accordingly.
"""

import torch
from typing import Dict


class GPUManager:
    """Manages GPU resources and optimizes batch sizes."""
    
    def __init__(self):
        self.device = self._detect_device()
        self.gpu_memory_gb = self._get_gpu_memory()
        self.batch_sizes = self._calculate_optimal_batches()
        
    def _detect_device(self) -> str:
        """Detect optimal device: CUDA vs CPU."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
        
    def _get_gpu_memory(self) -> float:
        """Get available GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / 1e9
        
    def _calculate_optimal_batches(self) -> Dict[str, int]:
        """Calculate optimal batch size for each attribution method."""
        base_sizes = {
            "saliency": 16,
            "inputxgradient": 16,
            "smoothgrad": 8,
            "guided_backprop": 16,
            "integrated_gradients": 4,
            "gradientshap": 2,
            "occlusion": 12,
            "xrai": 1,
            "grad_cam": 8,
            "guided_gradcam": 8,
            "random_baseline": 32,
            "c3f": 1,
        }
        
        # Adjust based on GPU memory
        if self.gpu_memory_gb > 16:
            return {k: v * 2 for k, v in base_sizes.items()}
        elif self.gpu_memory_gb > 8:
            return base_sizes
        else:
            return {k: max(1, v // 2) for k, v in base_sizes.items()}
            
    def get_batch_size(self, method: str) -> int:
        """Get optimal batch size for specific method."""
        return self.batch_sizes.get(method, 1)
        
    def get_strategy(self, method: str) -> str:
        """Get processing strategy for method."""
        if method in ["integrated_gradients", "gradientshap"]:
            return "micro"
        elif method in ["xrai", "c3f"]:
            return "single"
        else:
            return "batch"
            
    def print_info(self):
        """Print GPU information."""
        import logging
        logging.info(f"Device: {self.device}")
        if self.device == "cuda":
            logging.info(f"GPU Memory: {self.gpu_memory_gb:.1f}GB")
            logging.info(f"GPU Name: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Running on CPU - this will be slower")
        logging.info("=" * 60)

