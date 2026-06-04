import logging
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional

LayerExtractor = Callable[[nn.Module], nn.Module]


class TargetLayerRegistry:
    """Registry to map model architectures to their exact legacy CAM target layers."""

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

        if model_type in self._registry:
            return self._registry[model_type](model)

        logging.warning(f"Unknown architecture: {model_type}, falling back to find last Conv2d")
        last_conv = self._find_last_conv2d(model)
        if last_conv is not None:
            return last_conv

        raise NotImplementedError(
            f"Target layer selection for model type '{model_type}' is not implemented."
        )

    def _find_last_conv2d(self, model: nn.Module) -> Optional[nn.Module]:
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv


# Instantiate global registry
cam_layer_registry = TargetLayerRegistry()


@cam_layer_registry.register("EfficientNet")
def _get_efficientnet_layer(model: nn.Module) -> nn.Module:
    # Matches "features.8" exactly
    return model.features[8]

@cam_layer_registry.register("ResNet")
def _get_resnet_layer(model: nn.Module) -> nn.Module:
    return model.layer4

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
