"""
Architecture-specific utilities.

Handles layer selection for different model architectures.
"""

import torch.nn as nn
import logging


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
    model_type = type(model).__name__
    
    # ResNet family
    if hasattr(model, 'layer4') and hasattr(model.layer4, '__getitem__'):
        return model.layer4[-1]
    
    # VGG family
    elif hasattr(model, 'features') and isinstance(model.features, nn.Sequential):
        # Find last convolutional layer
        for layer in reversed(model.features):
            if isinstance(layer, nn.Conv2d):
                return layer
        return model.features[-1]
    
    # MobileNet family
    elif hasattr(model, 'features') and hasattr(model.features, '__getitem__'):
        return model.features[-1]
    
    # Vision Transformer (timm)
    elif hasattr(model, 'blocks') and hasattr(model.blocks, '__getitem__'):
        return model.blocks[-1].norm1
    
    # Swin Transformer (timm)
    elif hasattr(model, 'layers') and hasattr(model.layers, '__getitem__'):
        return model.layers[-1].blocks[-1].norm1
    
    else:
        logging.warning(f"Unknown architecture: {model_type}, trying to find last conv layer")
        # Generic fallback: try to find last conv layer
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        if last_conv is not None:
            return last_conv
        raise NotImplementedError(f"Layer selection for {model_type} not implemented")

