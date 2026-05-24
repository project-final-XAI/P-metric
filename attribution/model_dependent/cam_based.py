"""
CAM-based attribution methods.

Includes:
- GradCAM
- Guided GradCAM
"""

import torch
import torch.nn as nn

from attribution.base import AttributionMethod
from captum.attr import LayerGradCam, GuidedGradCam, LayerAttribution

# Import the registry we designed previously (assuming you put it in its own file or inside models/)
from models.architectures import cam_layer_registry


class GradCAMMethod(AttributionMethod):
    """GradCAM attribution."""

    def __init__(self):
        super().__init__("grad_cam")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Resolve layer cleanly
        target_layer = cam_layer_registry.get_target_layer(model)

        # 2. Compute attribution
        layer_gc = LayerGradCam(model, target_layer)
        attribution = layer_gc.attribute(images, target=targets, relu_attributions=True)

        # 3. Upsample to input resolution
        upsampled = LayerAttribution.interpolate(attribution, images.shape[2:], "bilinear")
        return upsampled


class GuidedGradCAMMethod(AttributionMethod):
    """Guided GradCAM attribution."""

    def __init__(self):
        super().__init__("guided_gradcam")

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Resolve layer cleanly
        target_layer = cam_layer_registry.get_target_layer(model)

        # 2. Compute attribution
        ggc = GuidedGradCam(model, target_layer)
        return ggc.attribute(images, target=targets)