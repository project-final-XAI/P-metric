"""
Perturbation-based attribution methods.

Includes:
- Occlusion
- XRAI (simplified implementation using IG)
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import Occlusion, IntegratedGradients


class OcclusionMethod(AttributionMethod):
    """Occlusion attribution."""

    def __init__(self):
        super().__init__("occlusion")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        occlusion = Occlusion(model)
        sliding_window_shapes = (3, 25, 25)
        strides = (3, 20, 20)
        return occlusion.attribute(
            images,
            target=targets,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=0,
        )


class XRAIMethod(AttributionMethod):
    """XRAI attribution using Integrated Gradients with region smoothing."""

    def __init__(self):
        super().__init__("xrai")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ig = IntegratedGradients(model)
        baseline = torch.zeros_like(images)
        attribution = ig.attribute(
            images,
            baselines=baseline,
            target=targets,
            n_steps=25,
        )
        # Take absolute values and aggregate across channels
        attribution = torch.abs(attribution)
        if attribution.ndim == 4:
            attribution = torch.mean(attribution, dim=1)
        # Apply smoothing for region-based effect
        return self._smooth_attribution(attribution)

    def _smooth_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Apply simple smoothing for region-based effect (vectorized)."""
        kernel_size = 5
        padding = kernel_size // 2
        if attribution.ndim == 2:
            attribution = attribution.unsqueeze(0).unsqueeze(0)
        elif attribution.ndim == 3:
            attribution = attribution.unsqueeze(1)
        smoothed = torch.nn.functional.avg_pool2d(
            attribution,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            count_include_pad=False,
        )
        if smoothed.shape[1] == 1:
            smoothed = smoothed.squeeze(1)
        return smoothed
