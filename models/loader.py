"""
Model loading utilities.

Handles loading pretrained models from torchvision and timm.
"""

import timm
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

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

        # If file doesn't exist, try alternative names (for SIPaK models with/without "cropped")
        if not os.path.exists(local_model_path) and 'sipak' in model_name.lower():
            # Try removing "cropped" from name
            alt_name = model_name.replace('_cropped', '').replace('cropped_', '')
            alt_path = os.path.join(parent_dir, 'models', alt_name)
            if os.path.exists(alt_path):
                local_model_path = alt_path
                logging.info(f"Found model with alternative name: {alt_name} (requested: {model_name})")
            else:
                # Try adding "cropped" to name
                if '_cropped' not in model_name:
                    alt_name2 = model_name.replace('.pth', '_cropped.pth')
                    alt_path2 = os.path.join(parent_dir, 'models', alt_name2)
                    if os.path.exists(alt_path2):
                        local_model_path = alt_path2
                        logging.info(f"Found model with alternative name: {alt_name2} (requested: {model_name})")

        if os.path.exists(local_model_path):
            try:
                checkpoint = torch.load(local_model_path, map_location=DEVICE, weights_only=False)
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    # Try different keys that might contain the model
                    if 'model' in checkpoint:
                        model = checkpoint['model']
                    elif 'state_dict' in checkpoint:
                        # If only state_dict, try to infer architecture from filename for SIPaK models
                        state_dict = checkpoint['state_dict']
                        if 'sipak' in model_name.lower() or 'efficientnet' in model_name.lower() or 'resnet' in model_name.lower():
                            # Infer architecture from filename
                            num_classes = checkpoint.get('num_classes', 5)  # SIPaK has 5 classes
                            
                            if 'efficientnet' in model_name.lower():
                                # Create EfficientNet-B0
                                model = models.efficientnet_b0(weights=None)
                                # Modify classifier for SIPaK (5 classes)
                                model.classifier = nn.Sequential(
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(model.classifier[1].in_features, num_classes)
                                )
                            elif 'resnet' in model_name.lower():
                                # Create ResNet50
                                model = models.resnet50(weights=None)
                                # Modify last layer for SIPaK (5 classes)
                                model.fc = nn.Linear(model.fc.in_features, num_classes)
                            else:
                                raise ValueError(f"Cannot infer architecture from filename: {model_name}")
                            
                            # Load state dict (strict=False to handle minor mismatches)
                            model.load_state_dict(state_dict, strict=False)
                            logging.info(f"Loaded SIPaK model {model_name} with {num_classes} classes from state_dict")
                        else:
                            raise ValueError(f"Checkpoint contains 'state_dict' but no model architecture info for {model_name}")
                    elif 'model_state_dict' in checkpoint:
                        # Same handling as state_dict
                        state_dict = checkpoint['model_state_dict']
                        if 'sipak' in model_name.lower() or 'efficientnet' in model_name.lower() or 'resnet' in model_name.lower():
                            num_classes = checkpoint.get('num_classes', 5)
                            
                            if 'efficientnet' in model_name.lower():
                                model = models.efficientnet_b0(weights=None)
                                model.classifier = nn.Sequential(
                                    nn.Dropout(p=0.2, inplace=True),
                                    nn.Linear(model.classifier[1].in_features, num_classes)
                                )
                            elif 'resnet' in model_name.lower():
                                model = models.resnet50(weights=None)
                                model.fc = nn.Linear(model.fc.in_features, num_classes)
                            else:
                                raise ValueError(f"Cannot infer architecture from filename: {model_name}")
                            
                            model.load_state_dict(state_dict, strict=False)
                            logging.info(f"Loaded SIPaK model {model_name} with {num_classes} classes from model_state_dict")
                        else:
                            raise ValueError(f"Checkpoint contains 'model_state_dict' but no model architecture info for {model_name}")
                    else:
                        # Unknown format - show available keys for debugging
                        raise ValueError(f"Unknown checkpoint format for {model_name}. Available keys: {list(checkpoint.keys())}")
                else:
                    # Checkpoint is the model itself (direct model object)
                    model = checkpoint
                    logging.info(f"Loaded model {model_name} directly (not a dict checkpoint)")
            except Exception as e1:
                raise ValueError(f"Failed to load local model '{local_model_path}': {e1}")
        else:
            raise ValueError(f"Local model file '{local_model_path}' does not exist.")
    else:
        try:
            # Try torchvision first
            model = models.get_model(model_name, weights="IMAGENET1K_V1")
        except Exception:
            try:
                # Try timm for ViT, Swin-T, etc.
                model = timm.create_model(model_name, pretrained=True)
            except Exception as e2:
                raise ValueError(f"Model '{model_name}' not found in torchvision, timm, or local models folder: {e2}")

    # Move to device (works for both CPU and GPU)
    model = model.to(DEVICE)

    # Prefer channels_last for CNNs to improve TensorCore utilization
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    model.eval()

    # Keep gradients enabled for attribution methods
    # Attribution methods need to compute gradients w.r.t. input
    for param in model.parameters():
        param.requires_grad = True

    # Model is ready for evaluation
    # Note: torch.compile is disabled for maximum compatibility

    return model
