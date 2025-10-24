"""
Gradient-based attribution methods.

Includes:
- Saliency (vanilla gradients)
- Input × Gradient
- SmoothGrad (averaged noisy gradients)
"""

import torch
import torch.nn as nn
from attribution.base import AttributionMethod
from captum.attr import Saliency, InputXGradient, NoiseTunnel


class SaliencyMethod(AttributionMethod):
    """Saliency attribution using vanilla gradients."""
    
    def __init__(self):
        super().__init__("saliency", "batch", 16)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute saliency attribution."""
        saliency = Saliency(model)
        attribution = saliency.attribute(images, target=targets)
        return self._normalize_attribution(attribution)


class InputXGradientMethod(AttributionMethod):
    """Input × Gradient attribution."""
    
    def __init__(self):
        super().__init__("inputxgradient", "batch", 16)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Input × Gradient attribution."""
        ixg = InputXGradient(model)
        attribution = ixg.attribute(images, target=targets)
        return self._normalize_attribution(attribution)


class SmoothGradMethod(AttributionMethod):
    """SmoothGrad attribution using noisy gradients."""
    
    def __init__(self):
        super().__init__("smoothgrad", "batch", 8)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute SmoothGrad attribution."""
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        attribution = nt.attribute(
            images, 
            nt_type='smoothgrad', 
            stdevs=0.1, 
            n_samples=10, 
            target=targets
        )
        return self._normalize_attribution(attribution)

