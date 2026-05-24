"""
Gradient-based attribution methods.

Includes:
- Saliency (vanilla gradients)
- Input × Gradient
- SmoothGrad (averaged noisy gradients)
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import Saliency, InputXGradient, NoiseTunnel


class SaliencyMethod(AttributionMethod):
    """Saliency attribution using vanilla gradients."""

    def __init__(self):
        super().__init__("saliency")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        saliency = Saliency(model)
        return saliency.attribute(images, target=targets)


class InputXGradientMethod(AttributionMethod):
    """Input × Gradient attribution."""

    def __init__(self):
        super().__init__("inputxgradient")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ixg = InputXGradient(model)
        return ixg.attribute(images, target=targets)


class SmoothGradMethod(AttributionMethod):
    """SmoothGrad attribution using noisy gradients."""

    def __init__(self):
        super().__init__("smoothgrad")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        saliency = Saliency(model)
        nt = NoiseTunnel(saliency)
        return nt.attribute(
            images,
            target=targets,
            nt_type='smoothgrad',
            nt_samples=10,
            stdevs=0.1,
        )
