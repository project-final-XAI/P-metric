"""
Perturbation-based attribution methods.

Includes:
- Occlusion
- XRAI (simplified implementation using IG)
"""

import torch
import numpy as np
from attribution.base import AttributionMethod
from captum.attr import Occlusion, IntegratedGradients


class OcclusionMethod(AttributionMethod):
    """Occlusion attribution."""
    
    def __init__(self):
        super().__init__("occlusion")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Occlusion attribution."""
        occlusion = Occlusion(model)
        sliding_window_shapes = (3, 25, 25)
        strides = (3, 20, 20)
        
        attribution = occlusion.attribute(
            images,
            target=targets,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=0
        )
        
        # No need to upsample, Occlusion already returns full resolution
        return self._normalize_attribution(attribution)


class XRAIMethod(AttributionMethod):
    """XRAI attribution using Integrated Gradients with segmentation."""
    
    def __init__(self):
        super().__init__("xrai")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute XRAI attribution.
        
        Simplified XRAI using Integrated Gradients as base with region aggregation.
        """
        # Use Integrated Gradients as base attribution method
        ig = IntegratedGradients(model)
        
        # Compute IG attribution with more steps for better quality
        baseline = torch.zeros_like(images)
        attribution = ig.attribute(
            images,
            baselines=baseline,
            target=targets,
            n_steps=25
        )
        
        # Take absolute values and aggregate across channels
        attribution = torch.abs(attribution)
        if attribution.ndim == 4:
            attribution = torch.mean(attribution, dim=1)
        
        # Apply smoothing for region-based effect (vectorized for batches)
        attribution = self._smooth_attribution(attribution)
        
        # Normalize (vectorized for batches)
        min_vals = attribution.view(attribution.shape[0], -1).min(dim=1, keepdim=True)[0]
        max_vals = attribution.view(attribution.shape[0], -1).max(dim=1, keepdim=True)[0]
        ranges = max_vals - min_vals
        ranges = ranges.view(attribution.shape[0], 1, 1)  # Reshape for broadcasting
        
        # Avoid division by zero
        ranges = torch.where(ranges > 0, ranges, torch.ones_like(ranges))
        normalized = (attribution - min_vals.view(attribution.shape[0], 1, 1)) / ranges
        
        return normalized
    
    def _smooth_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Apply simple smoothing for region-based effect (vectorized for batches)."""
        # Use PyTorch's avg_pool2d for efficient batch processing
        kernel_size = 5
        padding = kernel_size // 2
        
        # Add channel dimension for avg_pool2d (expects [B, C, H, W])
        if attribution.ndim == 2:
            attribution = attribution.unsqueeze(0).unsqueeze(0)
        elif attribution.ndim == 3:
            attribution = attribution.unsqueeze(1)
        
        # Apply average pooling with padding
        smoothed = torch.nn.functional.avg_pool2d(
            attribution, 
            kernel_size=kernel_size, 
            stride=1, 
            padding=padding,
            count_include_pad=False
        )
        
        # Remove channel dimension
        if smoothed.shape[1] == 1:
            smoothed = smoothed.squeeze(1)
        
        return smoothed
