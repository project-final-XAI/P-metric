"""
Core module for generating attribution maps (heatmaps) for a given model
and input. It uses the Captum library and is designed for modularity.
"""
from typing import Callable, Dict, Optional, Union


import saliency.core as saliency

import numpy as np
import torch
import torch.nn as nn
from captum.attr import (
    Saliency,
    IntegratedGradients,
    GuidedBackprop,
    InputXGradient,
    GuidedGradCam,
    NoiseTunnel,
    GradientShap,
    Occlusion,
    LayerGradCam,
    LayerAttribution,
)
from torchvision.models import ResNet
from transformers import ViTForImageClassification


def _normalize_attribution(attribution: torch.Tensor) -> np.ndarray:
    """Normalizes the attribution map to the [0, 1] range and converts to numpy."""
    # Move to CPU, convert to numpy, and take absolute values
    att_np = np.abs(attribution.squeeze().cpu().detach().numpy())

    # Handle single-channel or multi-channel heatmaps
    if att_np.ndim == 3:
        att_np = np.mean(att_np, axis=0)  # Average across channels if necessary

    # Normalize to [0, 1]
    min_val, max_val = np.min(att_np), np.max(att_np)
    if max_val > min_val:
        return (att_np - min_val) / (max_val - min_val)
    return att_np


def _get_target_layer(model: Union[ResNet, ViTForImageClassification]) -> nn.Module:
    """
    Intelligently selects a target layer for layer-based attribution methods
    like Grad-CAM. This is crucial for applying these methods to different
    architectures correctly.
    """
    if isinstance(model, ResNet):
        return model.layer4[-1]  # The last convolutional block in ResNet
    elif isinstance(model, ViTForImageClassification):
        # For Vision Transformers, the output of the last encoder layer is a good choice
        return model.vit.encoder.layer[-1].output
    # Add more elif blocks here for other architectures like Swin-T, EfficientNet etc.
    # elif isinstance(model, SwinTransformer): ...
    else:
        raise NotImplementedError(f"Layer selection for {type(model).__name__} is not implemented.")


# --- Original Attribution Methods ---

def _get_saliency_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Saliency attribution (gradient of output w.r.t. input)."""
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
    """Computes Layer Grad-CAM attribution using an intelligently selected layer."""
    target_layer = _get_target_layer(model)
    layer_gc = LayerGradCam(model, target_layer)
    # relu_attributions=True helps in focusing on features with a positive influence
    attribution = layer_gc.attribute(image, target, relu_attributions=True)
    upsampled_attribution = LayerAttribution.interpolate(attribution, image.shape[2:], "bilinear")
    return _normalize_attribution(upsampled_attribution)


def _get_input_x_gradient_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Input x Gradient attribution."""
    ixg = InputXGradient(model)
    attribution = ixg.attribute(image, target=target)
    return _normalize_attribution(attribution)


def _get_guided_grad_cam_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """Computes Guided Grad-CAM, combining Guided Backprop with Grad-CAM for finer detail."""
    target_layer = _get_target_layer(model)
    ggc = GuidedGradCam(model, target_layer)
    attribution = ggc.attribute(image, target)
    return _normalize_attribution(attribution)


def _get_smoothgrad_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """
    Computes SmoothGrad attribution by averaging gradients from noisy versions of the input.
    This helps to reduce noise in the attribution map. It's a wrapper around another method.
    """
    # SmoothGrad is implemented in Captum using NoiseTunnel, which wraps another attribution algorithm.
    # Here, we wrap Saliency, but it could also be Integrated Gradients, etc.
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    attribution = nt.attribute(image, nt_type='smoothgrad', stdevs=0.1, n_samples=10, target=target)
    return _normalize_attribution(attribution)


def _get_gradient_shap_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """
    Computes GradientSHAP attribution. It approximates SHAP values using expected gradients.
    Requires a baseline (e.g., a black image) to compare against.
    """
    # Create a baseline distribution of images. Here, we use a single black image.
    baseline = torch.zeros_like(image)
    gs = GradientShap(model)
    # n_samples determines how many random points along the path from baseline to input are sampled.
    attribution = gs.attribute(image, baselines=baseline, n_samples=50, stdevs=0.0, target=target)
    return _normalize_attribution(attribution)


def _get_xrai_attribution(model, image, target):
    def call_model_function(images, call_model_args=None):
        with torch.no_grad():
            preds = model(images)
        return preds.cpu().numpy()

    ig = saliency.IntegratedGradients()
    xrai = saliency.XRAI()

    baseline = np.zeros_like(image.cpu().numpy())
    ig_attributions = ig.GetMask(image.cpu().numpy(), call_model_function,
                                 call_model_args={'target': target},
                                 x_steps=25, x_baseline=baseline)

    xrai_attributions = xrai.GetMask(image.cpu().numpy(), call_model_function,
                                     call_model_args={'target': target},
                                     baselines=baseline)

    return xrai_attributions


def _get_occlusion_attribution(model: nn.Module, image: torch.Tensor, target: int) -> np.ndarray:
    """
    Computes Occlusion attribution, a perturbation-based method that measures the
    drop in prediction confidence when parts of the image are masked.
    """
    occlusion = Occlusion(model)
    # Define the size of the sliding window to occlude parts of the image.
    # The first dimension (3) should match the number of image channels.
    sliding_window_shapes = (3, 15, 15)
    # Define the step size for the sliding window.
    strides = (3, 8, 8)
    attribution = occlusion.attribute(image,
                                      strides=strides,
                                      target=target,
                                      sliding_window_shapes=sliding_window_shapes,
                                      baselines=0)  # Use a black baseline for occlusion
    # Occlusion attribution needs to be upsampled to match the original image size.
    upsampled_attribution = LayerAttribution.interpolate(attribution, image.shape[2:], "bilinear")
    return _normalize_attribution(upsampled_attribution)


# --- Dispatcher Dictionary ---
# This registry allows for easy extension. To add a new method:
# 1. Write a new `_get_..._attribution` function above.
# 2. Add its user-facing name and the function handle to this dictionary.
ATTRIBUTION_REGISTRY: Dict[str, Callable[[nn.Module, torch.Tensor, int], np.ndarray]] = {
    "saliency": _get_saliency_attribution,
    "integrated_gradients": _get_integrated_gradients_attribution,
    "guided_backprop": _get_guided_backprop_attribution,
    "grad_cam": _get_grad_cam_attribution,
    "saliency_mask": _get_saliency_attribution,
    "integrated_gradients_mask": _get_integrated_gradients_attribution,
    "guided_backprop_mask": _get_guided_backprop_attribution,
    "gradcam_mask_once": _get_grad_cam_attribution,
    "vit_gradcam_token": _get_grad_cam_attribution,
    "inputxgradient_mask": _get_input_x_gradient_attribution,
    "guided_gradcam_mask": _get_guided_grad_cam_attribution,
    "smoothgrad_mask": _get_smoothgrad_attribution,
    "gradientshap_mask": _get_gradient_shap_attribution,
    "xrai_mask": _get_xrai_attribution,
    "occlusion_mask": _get_occlusion_attribution,
    "naive_occ_mask": _get_occlusion_attribution,
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
        # Retrieve the function from the registry and call it
        attribution_func = ATTRIBUTION_REGISTRY[method_name]
        heatmap = attribution_func(model, image, target_class)
        return heatmap
    except Exception as e:
        print(f"Error generating attribution for method '{method_name}' on model '{type(model).__name__}': {e}")
        print("This might happen if a method is incompatible with the model architecture. Skipping.")
        return None
