"""
Other attribution methods.

Includes:
- Guided Backpropagation
- Random Baseline
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import GuidedBackprop


class GuidedBackpropMethod(AttributionMethod):
    """Guided Backpropagation attribution."""
    
    def __init__(self):
        super().__init__("guided_backprop")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Guided Backpropagation attribution."""
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(images, target=targets)
        return self._normalize_attribution(attribution)


class RandomBaselineMethod(AttributionMethod):
    """Random baseline attribution."""
    
    def __init__(self):
        super().__init__("random_baseline")
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute random baseline attribution."""
        return torch.rand(images.shape[0], images.shape[2], images.shape[3])

