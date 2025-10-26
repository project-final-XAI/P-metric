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
        super().__init__("occlusion", "batch", 8)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Occlusion attribution."""
        occlusion = Occlusion(model)
        sliding_window_shapes = (3, 20, 20)
        strides = (3, 16, 16)
        
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
        super().__init__("xrai", "single", 1)
        
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
            n_steps=50
        )
        
        # Take absolute values and aggregate across channels
        attribution = torch.abs(attribution)
        if attribution.ndim == 4:
            attribution = torch.mean(attribution, dim=1)
        
        # Apply smoothing for region-based effect
        attribution = self._smooth_attribution(attribution)
        
        # Normalize
        normalized = []
        for attr in attribution:
            min_val, max_val = attr.min(), attr.max()
            if max_val > min_val:
                normalized.append((attr - min_val) / (max_val - min_val))
            else:
                normalized.append(attr)
        
        return torch.stack(normalized)
    
    def _smooth_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Apply simple smoothing for region-based effect."""
        # Simple 3x3 average pooling for smoothing
        smoothed = []
        for attr in attribution:
            # Convert to numpy for easier manipulation
            attr_np = attr.cpu().detach().numpy()
            
            # Apply simple moving average
            kernel_size = 5
            pad = kernel_size // 2
            padded = np.pad(attr_np, pad, mode='edge')
            
            result = np.zeros_like(attr_np)
            for i in range(attr_np.shape[0]):
                for j in range(attr_np.shape[1]):
                    result[i, j] = np.mean(
                        padded[i:i+kernel_size, j:j+kernel_size]
                    )
            
            smoothed.append(torch.from_numpy(result))
        
        return torch.stack(smoothed)
