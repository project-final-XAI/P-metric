import logging
import torch.nn as nn
from typing import Callable, Dict, Optional

# Define a type for our extraction functions
LayerExtractor = Callable[[nn.Module], nn.Module]

class TargetLayerRegistry:
    """Registry to map model architectures to their CAM target layers."""
    
    def __init__(self):
        self._registry: Dict[str, LayerExtractor] = {}

    def register(self, model_type_name: str) -> Callable:
        """Decorator to register a new extraction strategy."""
        def decorator(func: LayerExtractor):
            self._registry[model_type_name] = func
            return func
        return decorator

    def get_target_layer(self, model: nn.Module) -> nn.Module:
        """Find the target layer, routing to the specific strategy."""
        model_type = type(model).__name__
        
        # 1. Check if we have a registered strategy for this specific model
        if model_type in self._registry:
            return self._registry[model_type](model)
            
        # 2. Generic fallback (The "Good" part of your original code)
        logging.warning(f"Unknown architecture: {model_type}, falling back to find last Conv2d")
        last_conv = self._find_last_conv2d(model)
        if last_conv is not None:
            return last_conv
            
        raise NotImplementedError(
            f"Target layer selection for model type '{model_type}' is not implemented."
        )

    def _find_last_conv2d(self, model: nn.Module) -> Optional[nn.Module]:
        """Recursively find the last Conv2d layer."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

# Instantiate the global registry
cam_layer_registry = TargetLayerRegistry()

# --- Now you can add support for models ANYWHERE in your codebase ---

@cam_layer_registry.register("ResNet")
def _get_resnet_layer(model: nn.Module) -> nn.Module:
    return model.layer4[-1]

@cam_layer_registry.register("VisionTransformer")
def _get_vit_layer(model: nn.Module) -> nn.Module:
    return model.blocks[-1].norm1

@cam_layer_registry.register("VGG")
def _get_vgg_layer(model: nn.Module) -> nn.Module:
    features = model.features
    for layer in reversed(features):
        if isinstance(layer, nn.Conv2d):
            return layer
    return features[-1]

@cam_layer_registry.register("EfficientNet")
def _get_efficientnet_layer(model: nn.Module) -> nn.Module:
    """
    Extracts the final convolutional layer for torchvision's EfficientNet.
    """
    try:
        return model.features[-1][0]
    except (AttributeError, IndexError) as e:
        import logging
        logging.error(f"Failed to extract EfficientNet target layer: {e}")
        raise