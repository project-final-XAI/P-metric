"""
Perturbation-based attribution methods.

Includes:
- Occlusion
- XRAI
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import Occlusion
from captum.attr import LayerAttribution


class OcclusionMethod(AttributionMethod):
    """Occlusion attribution."""
    
    def __init__(self):
        super().__init__("occlusion", "batch", 8)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Occlusion attribution."""
        occlusion = Occlusion(model)
        sliding_window_shapes = (3, 15, 15)
        strides = (3, 8, 8)
        
        attribution = occlusion.attribute(
            images,
            strides=strides,
            target=targets,
            sliding_window_shapes=sliding_window_shapes,
            baselines=0
        )
        
        upsampled = LayerAttribution.interpolate(attribution, images.shape[2:], "bilinear")
        return self._normalize_attribution(upsampled)


class XRAIMethod(AttributionMethod):
    """XRAI attribution."""
    
    def __init__(self):
        super().__init__("xrai", "single", 1)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute XRAI attribution (single image only)."""
        # XRAI implementation would go here
        # For now, return random baseline as placeholder
        return torch.rand(images.shape[0], images.shape[2], images.shape[3])
