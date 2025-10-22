"""
Core module for generating attribution maps (heatmaps) for a given model
and input, now fully supporting **Batch Processing** for high GPU utilization.
"""
from typing import Callable, Dict, Optional, Union, List
from torchvision.models import ResNet, MobileNetV2
from transformers import ViTForImageClassification
# ...

import saliency.core as saliency # Note: saliency.core might need special handling for batches

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

def _normalize_attribution(attribution_batch: torch.Tensor) -> torch.Tensor:
    """
    Normalizes a batch of attribution maps to the [0, 1] range.

    Args:
        attribution_batch: Tensor of shape [B, C, H, W] or [B, H, W] from Captum.

    Returns:
        Tensor of shape [B, H, W] normalized to [0, 1].
    """
    # 1. Take absolute values
    att_abs = torch.abs(attribution_batch.cpu().detach())

    # 2. Handle multi-channel heatmaps: Average across channels
    if att_abs.ndim == 4 and att_abs.shape[1] > 1:
        att_np = torch.mean(att_abs, dim=1)  # Result shape: [B, H, W]
    elif att_abs.ndim == 4 and att_abs.shape[1] == 1:
        att_np = att_abs.squeeze(1)  # Result shape: [B, H, W]
    elif att_abs.ndim == 3:
        att_np = att_abs # Result shape: [B, H, W]
    else:
        raise ValueError(f"Unexpected attribution tensor shape: {att_abs.shape}")

    # 3. Normalize each heatmap in the batch independently to [0, 1]
    normalized_heatmaps = []

    # Normalization needs to happen across each BATCH ELEMENT (image) independently
    # Iterating over the batch is necessary here unless normalization is by global min/max
    for heatmap in att_np:
        min_val, max_val = heatmap.min(), heatmap.max()
        if max_val > min_val:
            normalized_map = (heatmap - min_val) / (max_val - min_val)
        else:
            normalized_map = heatmap # Avoid division by zero
        normalized_heatmaps.append(normalized_map)

    return torch.stack(normalized_heatmaps) # Shape: [B, H, W]


def _get_target_layer(model: Union[ResNet, ViTForImageClassification, MobileNetV2]) -> nn.Module:
    """
    Intelligently selects a target layer for layer-based attribution methods
    like Grad-CAM.
    """

    if isinstance(model, ResNet):
        return model.layer4[-1]  # The last convolutional block in ResNet
    elif isinstance(model, ViTForImageClassification):
        return model.vit.encoder.layer[-1].output
    elif isinstance(model, MobileNetV2):
        # MobileNetV2 features are in the 'features' sequential module.
        # The last block before the final classifier.
        # This typically points to the last Convolutional layer (Conv2d).
        return model.features[-1]
    else:
        raise NotImplementedError(f"Layer selection for {type(model).__name__} is not implemented.")


# image: [B, C, H, W], target: [B] -> returns attribution: [B, C, H, W] or [B, H, W]
AttributionFunc = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


def _get_saliency_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Saliency attribution (gradient of output w.r.t. input) for a batch."""
    saliency = Saliency(model)
    # Target must be a 1D tensor of class indices for Captum batching
    attribution_batch = saliency.attribute(image, target=target)
    return _normalize_attribution(attribution_batch)


def _get_integrated_gradients_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> Optional[
    torch.Tensor]:
    """
    Computes Integrated Gradients attribution. Processes one image at a time
    to mitigate VRAM overflow (since IG is highly memory-intensive).
    """
    n_steps = 10

    ig = IntegratedGradients(model)
    all_attributions = []

    for i in range(image.shape[0]):
        single_image = image[i:i + 1]  # Batch size 1
        single_target = target[i:i + 1]  # Batch size 1

        # Core Attribution Logic
        try:
            attribution_single = ig.attribute(
                single_image,
                target=single_target,
                n_steps=n_steps,
                baselines=torch.zeros_like(single_image)
            )
            all_attributions.append(attribution_single)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(
                    f"Extreme Warning: Integrated Gradients (B=1) failed at step {n_steps}. Consider reducing n_steps further.")
                return None
            else:
                raise e

    # 3. איחוד ונירמול
    combined_attributions = torch.cat(all_attributions, dim=0)
    return _normalize_attribution(combined_attributions)


def _get_guided_backprop_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Guided Backpropagation attribution for a batch."""
    gbp = GuidedBackprop(model)
    attribution_batch = gbp.attribute(image, target=target)
    return _normalize_attribution(attribution_batch)


def _get_grad_cam_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> Optional[torch.Tensor]:
    """Computes Layer Grad-CAM attribution for a batch, including error handling."""
    was_training = model.training
    model.eval()

    try:
        target_layer = _get_target_layer(model)
        layer_gc = LayerGradCam(model, target_layer)

        try:
            attribution_batch = layer_gc.attribute(image, target, relu_attributions=True)

        except RuntimeError as e:
            if "not have been used in the graph" in str(e) or "allow_unused=True" in str(e):
                print(
                    f"Warning: Grad-CAM failed for {type(model).__name__} due to 'allow_unused' error. Skipping this batch.")
                return None
            else:
                raise e

        if attribution_batch is None:
            return None

        upsampled_attribution = LayerAttribution.interpolate(attribution_batch, image.shape[2:], "bilinear")
        return _normalize_attribution(upsampled_attribution)

    except Exception as e:
        print(f"Error during Grad-CAM attribution: {e}")
        return None

    finally:
        if was_training:
            model.train()  # החזרת המצב הקודם


def _get_input_x_gradient_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Input x Gradient attribution for a batch."""
    ixg = InputXGradient(model)
    attribution_batch = ixg.attribute(image, target=target)
    return _normalize_attribution(attribution_batch)


def _get_guided_grad_cam_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes Guided Grad-CAM for a batch."""
    target_layer = _get_target_layer(model)
    ggc = GuidedGradCam(model, target_layer)
    attribution_batch = ggc.attribute(image, target)
    return _normalize_attribution(attribution_batch)


def _get_smoothgrad_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes SmoothGrad attribution for a batch."""
    saliency = Saliency(model)
    nt = NoiseTunnel(saliency)
    # Captum's NoiseTunnel supports batch input when wrapping an attribution method that supports it.
    attribution_batch = nt.attribute(image, nt_type='smoothgrad', stdevs=0.1, n_samples=10, target=target)
    return _normalize_attribution(attribution_batch)


def _get_gradient_shap_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes GradientSHAP attribution. If the input is a batch (B > 1),
    it will process one image at a time to mitigate VRAM overflow (since
    GradientSHAP is highly memory-intensive).
    """

    # --- Extreme VRAM Management for GradientSHAP ---

    # 1. Configuration
    n_samples = 5  # Keep this low for memory

    # 2. Iterate over the batch (B must be 1 here to survive on 8GB VRAM)
    all_attributions = []

    for i in range(image.shape[0]):
        single_image = image[i:i + 1]  # Take one image (B=1)
        single_target = target[i:i + 1]  # Take one target (B=1)

        # Core Attribution Logic
        baseline = torch.zeros_like(single_image)
        gs = GradientShap(model)  # Re-initialize if necessary, or just call attribute

        try:
            attribution_single = gs.attribute(
                single_image,
                baselines=baseline,
                n_samples=n_samples,
                stdevs=0.0,
                target=single_target
            )
            all_attributions.append(attribution_single)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print("Extreme Warning: GradientSHAP (B=1) still failed. Consider reducing n_samples further.")
                # Return None for this batch to allow the process to continue
                return None
            else:
                raise e

    # 3. Combine results back into a batch
    combined_attributions = torch.cat(all_attributions, dim=0)

    return _normalize_attribution(combined_attributions)


def _get_occlusion_attribution(model: nn.Module, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes Occlusion attribution for a batch.
    """
    occlusion = Occlusion(model)
    # Parameters for the sliding window
    sliding_window_shapes = (3, 15, 15)
    strides = (3, 8, 8)
    # The `attribute` call handles batching internally if the window parameters are defined correctly
    attribution_batch = occlusion.attribute(image,
                                      strides=strides,
                                      target=target,
                                      sliding_window_shapes=sliding_window_shapes,
                                      baselines=0)
    # Occlusion attribution needs to be upsampled to match the original image size.
    upsampled_attribution = LayerAttribution.interpolate(attribution_batch, image.shape[2:], "bilinear")
    return _normalize_attribution(upsampled_attribution)


# --- Dispatcher Dictionary ---
# Update registry type to match the new Batch-supporting function signature
ATTRIBUTION_REGISTRY: Dict[str, AttributionFunc] = {
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
    "occlusion_mask": _get_occlusion_attribution,
    "naive_occ_mask": _get_occlusion_attribution,
}

def generate_attribution(
        model: nn.Module,
        image: torch.Tensor,
        target_class: torch.Tensor,
        method_name: str
) -> Optional[torch.Tensor]:
    """
    Generates a batch of attribution maps (heatmaps) for a given model and input batch.

    Args:
        model: The model to generate attribution for.
        image: The input image tensor (Batch, shape [B, C, H, W]).
        target_class: The target class indices for the attribution (Tensor, shape [B]).
        method_name: The name of the attribution method.

    Returns:
        A BxHxW tensor representing the normalized heatmaps batch, or None.
    """
    if method_name not in ATTRIBUTION_REGISTRY:
        print(f"Warning: Attribution method '{method_name}' not recognized. Skipping.")
        return None

    try:
        # Retrieve the function from the registry and call it with the batch
        attribution_func: AttributionFunc = ATTRIBUTION_REGISTRY[method_name]

        # heatmap_batch should be [B, H, W]
        heatmap_batch = attribution_func(model, image, target_class)
        return heatmap_batch
    except Exception as e:
        # Note: If XRAI is needed, you must implement its batching logic separately.
        print(f"Error generating attribution for method '{method_name}' on model '{type(model).__name__}': {e}")
        print("Ensure the underlying attribution method supports batch input (B > 1) and try again. Skipping.")
        return None