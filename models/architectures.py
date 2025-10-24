"""
Architecture-specific utilities.

Handles layer selection for different model architectures.
"""

import torch.nn as nn
from torchvision.models import ResNet, MobileNetV2
from transformers import ViTForImageClassification


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Get target layer for CAM-based methods.
    
    Args:
        model: Neural network model
        
    Returns:
        Target layer for attribution
        
    Raises:
        NotImplementedError: If architecture not supported
    """
    if isinstance(model, ResNet):
        return model.layer4[-1]  # Last conv block
    elif isinstance(model, MobileNetV2):
        return model.features[-1]  # Last feature layer
    elif isinstance(model, ViTForImageClassification):
        return model.vit.encoder.layer[-1].output  # Last encoder layer
    else:
        # Try to get from timm models
        if hasattr(model, 'blocks') and hasattr(model.blocks, 'layer'):
            return model.blocks.layer[-1]  # Swin-T style
        elif hasattr(model, 'layers') and hasattr(model.layers, 'layer'):
            return model.layers.layer[-1]  # Other transformer style
        else:
            raise NotImplementedError(f"Layer selection for {type(model).__name__} not implemented")

