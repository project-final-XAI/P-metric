"""
CAM-based attribution methods.

Includes:
- GradCAM
- Guided GradCAM
"""

import torch
from attribution.base import AttributionMethod
from captum.attr import LayerGradCam, GuidedGradCam, LayerAttribution
# Import will be handled dynamically to avoid circular imports


class GradCAMMethod(AttributionMethod):
    """GradCAM attribution."""
    
    def __init__(self):
        super().__init__("grad_cam", "batch", 8)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute GradCAM attribution."""
        from models.architectures import get_target_layer
        target_layer = get_target_layer(model)
        layer_gc = LayerGradCam(model, target_layer)
        
        attribution = layer_gc.attribute(images, targets, relu_attributions=True)
        upsampled = LayerAttribution.interpolate(attribution, images.shape[2:], "bilinear")
        return self._normalize_attribution(upsampled)


class GuidedGradCAMMethod(AttributionMethod):
    """Guided GradCAM attribution."""
    
    def __init__(self):
        super().__init__("guided_gradcam", "batch", 8)
        
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Guided GradCAM attribution."""
        from models.architectures import get_target_layer
        target_layer = get_target_layer(model)
        ggc = GuidedGradCam(model, target_layer)
        attribution = ggc.attribute(images, targets)
        return self._normalize_attribution(attribution)
