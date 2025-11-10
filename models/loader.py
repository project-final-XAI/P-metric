"""
Model loading utilities.

Handles loading pretrained models from torchvision and timm.
"""

import timm
import torch.nn as nn
import torchvision.models as models
from config import DEVICE, USE_TORCH_COMPILE

import os
import torch
import logging


def load_model(model_name: str) -> nn.Module:
    """
    Load a specified model and prepare it for evaluation.

    Args:
        model_name: Model name (e.g., 'resnet50', 'vit_b_16', 'swin_t')

    Returns:
        PyTorch model in evaluation mode on configured device

    Raises:
        ValueError: If model not found in torchvision, timm, or local models folder
    """
    if model_name.endswith('.pth'):
        # Go up one level from models/ folder to project root, then into models/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        local_model_path = os.path.join(parent_dir, 'models', model_name)

        if os.path.exists(local_model_path):
            try:
                checkpoint = torch.load(local_model_path, map_location=DEVICE, weights_only=False)
                model = checkpoint['model']
                # logging.info(f"Loaded {model_name} from local file")
            except Exception as e1:
                raise ValueError(f"Failed to load local model '{local_model_path}': {e1}")
        else:
            raise ValueError(f"Local model file '{local_model_path}' does not exist.")
    else:
        try:
            # Try torchvision first
            model = models.get_model(model_name, weights="IMAGENET1K_V1")
            # logging.info(f"Loaded {model_name} from torchvision")
        except Exception:
            try:
                # Try timm for ViT, Swin-T, etc.
                model = timm.create_model(model_name, pretrained=True)
                # logging.info(f"Loaded {model_name} from timm")
            except Exception as e2:
                raise ValueError(f"Model '{model_name}' not found in torchvision, timm, or local models folder: {e2}")

    # Move to device (works for both CPU and GPU)
    model = model.to(DEVICE)

    model.eval()

    # Keep gradients enabled for attribution methods
    # Attribution methods need to compute gradients w.r.t. input
    for param in model.parameters():
        param.requires_grad = True

    # Apply torch.compile for optimization (PyTorch 2.0+)
    if USE_TORCH_COMPILE and DEVICE == "cuda":
        try:
            # Use 'reduce-overhead' mode for optimal inference performance
            # This enables CUDA graphs and other optimizations
            model = torch.compile(model, mode='reduce-overhead')
            logging.debug(f"Successfully compiled {model_name} with torch.compile")
        except Exception as e:
            # Fallback gracefully if compilation fails
            logging.warning(f"torch.compile failed for {model_name}, using eager mode: {e}")

    return model
