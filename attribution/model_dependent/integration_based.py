"""
Integration-based attribution methods.

Includes:
- Integrated Gradients
- GradientSHAP
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import IntegratedGradients, GradientShap


class IntegratedGradientsMethod(AttributionMethod):
    """Integrated Gradients attribution."""
    
    def __init__(self):
        super().__init__("integrated_gradients")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Integrated Gradients attribution."""
        ig = IntegratedGradients(model)
        attribution = ig.attribute(
            images, 
            target=targets, 
            n_steps=6,
            baselines=torch.zeros_like(images),
            internal_batch_size=32  # Increased for better VRAM utilization
        )
        return self._normalize_attribution(attribution)


class GradientSHAPMethod(AttributionMethod):
    """GradientSHAP attribution."""
    
    def __init__(self):
        super().__init__("gradientshap")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute GradientSHAP attribution."""
        baseline = torch.zeros_like(images)
        gs = GradientShap(model)
        attribution = gs.attribute(
            images,
            baselines=baseline,
            target=targets,
            n_samples=4  # Increased for better VRAM utilization
        )
        return self._normalize_attribution(attribution)

