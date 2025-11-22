"""
Base class for all attribution methods.

Defines unified interface for computing attribution maps.
Batch sizing is handled by GPUManager, not by individual methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union
import torch
import numpy as np
import logging


class AttributionMethod(ABC):
    """Base class for all attribution methods."""
    
    def __init__(self, name: str):
        """
        Initialize attribution method.
        
        Args:
            name: Unique identifier for this attribution method
        """
        self.name = name
        
    @abstractmethod
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute attribution maps for a batch of images.
        
        Args:
            model: Neural network model
            images: Batch of images (B, C, H, W)
            targets: Target class indices (B,)
            
        Returns:
            Attribution heatmaps (B, H, W) normalized to [0, 1]
        """
        pass
        
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

