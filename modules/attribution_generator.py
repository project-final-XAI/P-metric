# modules/attribution_generator.py
"""
Core module for generating attribution maps (heatmaps) for a given model
and input. It uses the Captum library and is designed for modularity.
"""
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    LayerGradCam,
    LayerAttribution
)
from torchvision.models import ResNet
from transformers import ViTForImageClassification


def _normalize_attribution(attribution: torch.Tensor) -> np.ndarray:
    """Normalizes the attribution map to the [0, 1] range and converts to numpy."""
    # Move to CPU, convert to numpy, and take absolute values as per the paper's method for some techniques
    att_np = np.abs(attribution.squeeze().cpu().detach().numpy())

    # Handle single-channel or multi-channel heatmaps
    if att_np.ndim == 3:
        att_np = np.mean(att_np, axis=0)  # Average across channels if necessary

    # Normalize to [0, 1]
    min_val, max_val = np.min(att_np), np.max(att_np)
    if max_val > min_val:
        return (att_np - min_val) / (max_val - min_val)
    return att_np


def _get_saliency_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Saliency attribution."""
    saliency = Saliency(model)
    attribution = saliency.attribute(image, target=target)
    return _normalize_attribution(attribution)


def _get_integrated_gradients_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Integrated Gradients attribution."""
    ig = IntegratedGradients(model)
    attribution = ig.attribute(image, target=target, n_steps=50, internal_batch_size=1)
    return _normalize_attribution(attribution)


def _get_guided_backprop_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Guided Backpropagation attribution."""
    gbp = GuidedBackprop(model)
    attribution = gbp.attribute(image, target=target)
    return _normalize_attribution(attribution)


def _get_grad_cam_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """
    Computes Layer Grad-CAM attribution.

    This function intelligently selects the target layer based on the model
    architecture (CNN vs. Transformer), as different architectures require
    different handling.
    """
    # 1. Select the target layer based on model architecture
    if isinstance(model, ResNet):
        target_layer = model.layer4[-1]  # Last convolutional block
    elif isinstance(model, ViTForImageClassification):
        # As per pytorch-grad-cam library recommendations for ViT
        target_layer = model.vit.encoder.layer[-1].output
    # Add more elif blocks here for other architectures like Swin-T, EfficientNet etc.
    # elif isinstance(model, SwinTransformer): ...
    else:
        raise NotImplementedError(f"Grad-CAM layer selection for {type(model).__name__} is not implemented.")

    # 2. Compute Grad-CAM
    layer_gc = LayerGradCam(model, target_layer)
    attribution = layer_gc.attribute(image, target, relu_attributions=True)

    # 3. Upsample the heatmap to match the input image size
    # LayerAttribution.interpolate is a convenient method in Captum for this
    upsampled_attribution = LayerAttribution.interpolate(attribution, image.shape[2:], "bilinear")

    return _normalize_attribution(upsampled_attribution)


# --- Dispatcher Dictionary ---
# This is the key to modularity. To add a new method:
# 1. Write a new `_get_..._attribution` function.
# 2. Add its name and function handle to this dictionary.
ATTRIBUTION_REGISTRY: Dict[str, Callable[[nn.Module, torch.Tensor, int], np.ndarray]] = {
    "saliency": _get_saliency_attribution,
    "integrated_gradients": _get_integrated_gradients_attribution,
    "guided_backprop": _get_guided_backprop_attribution,
    "grad_cam": _get_grad_cam_attribution,
}


def generate_attribution(
        model: nn.Module,
        image: torch.Tensor,
        target_class: int,
        method_name: str
) -> Optional[np.ndarray]:
    """
    Generates an attribution map for a given image and model using the specified method.

    This is the main entry point for this module. It uses the dispatcher
    pattern to call the correct underlying attribution function.

    Args:
        model: The model to generate attribution for.
        image: The input image tensor (should be a single image with batch dim).
        target_class: The target class index for the attribution.
        method_name: The name of the attribution method (must be in ATTRIBUTION_REGISTRY).

    Returns:
        A 2D numpy array representing the normalized heatmap, or None if the
        method fails or is not supported.
    """
    if method_name not in ATTRIBUTION_REGISTRY:
        print(f"Warning: Attribution method '{method_name}' not recognized. Skipping.")
        return None

    if image.shape[0] != 1:
        raise ValueError(f"This function expects a single image, but got batch size {image.shape[0]}")

    try:
        # print(f"Generating '{method_name}' attribution...")
        # Retrieve the function from the registry and call it
        attribution_func = ATTRIBUTION_REGISTRY[method_name]
        heatmap = attribution_func(model, image, target_class)
        # print(f"Successfully generated '{method_name}' heatmap.")
        return heatmap
    except Exception as e:
        print(f"Error generating attribution for method '{method_name}' on model '{type(model).__name__}': {e}")
        print("This might happen if a method is incompatible with the model architecture. Skipping.")
        return None
