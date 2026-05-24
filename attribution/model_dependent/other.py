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

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        gbp = GuidedBackprop(model)
        return gbp.attribute(images, target=targets)


class RandomBaselineMethod(AttributionMethod):
    """Random baseline attribution (control)."""

    def __init__(self):
        super().__init__("random_baseline")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.rand(images.shape[0], images.shape[2], images.shape[3])
