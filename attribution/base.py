"""
Base class for all attribution methods.

Defines unified interface and Adapter Pattern for different
processing strategies (batch, micro-batch, single).
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import torch
import numpy as np
import logging


class AttributionMethod(ABC):
    """Base class for all attribution methods."""
    
    def __init__(self, name: str, strategy: str, max_batch_size: int):
        self.name = name
        self.strategy = strategy  # "batch" | "micro" | "single"
        self.max_batch_size = max_batch_size
        
    @abstractmethod
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute attribution maps."""
        pass
        
    def __call__(self, model, images: torch.Tensor, targets: torch.Tensor) -> Optional[torch.Tensor]:
        """Adapter method with automatic fallback strategy."""
        try:
            if self.strategy == "batch":
                return self.compute(model, images, targets)
            elif self.strategy == "micro":
                return self._micro_batch(model, images, targets)
            else:  # single
                return self._single_image(model, images, targets)
        except Exception as e:
            logging.error(f"Error in {self.name}: {e}")
            return None
            
    def _micro_batch(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Process in micro-batches to avoid memory issues."""
        results = []
        batch_size = min(self.max_batch_size, images.shape[0])
        
        for i in range(0, images.shape[0], batch_size):
            end_idx = min(i + batch_size, images.shape[0])
            batch_images = images[i:end_idx]
            batch_targets = targets[i:end_idx]
            
            batch_result = self.compute(model, batch_images, batch_targets)
            if batch_result is not None:
                results.append(batch_result)
                
        if results:
            return torch.cat(results, dim=0)
        return None
        
    def _single_image(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Process one image at a time."""
        results = []
        
        for i in range(images.shape[0]):
            single_image = images[i:i+1]
            single_target = targets[i:i+1]
            
            result = self.compute(model, single_image, single_target)
            if result is not None:
                results.append(result)
                
        if results:
            return torch.cat(results, dim=0)
        return None
        
    def _normalize_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Normalize attribution to [0, 1] range."""
        # Take absolute values
        att_abs = torch.abs(attribution.cpu().detach())
        
        # Handle multi-channel heatmaps
        if att_abs.ndim == 4 and att_abs.shape[1] > 1:
            att_abs = torch.mean(att_abs, dim=1)
        elif att_abs.ndim == 4 and att_abs.shape[1] == 1:
            att_abs = att_abs.squeeze(1)
            
        # Normalize each heatmap independently
        normalized = []
        for heatmap in att_abs:
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val > min_val:
                normalized.append((heatmap - min_val) / (max_val - min_val))
            else:
                normalized.append(heatmap)
                
        return torch.stack(normalized)

