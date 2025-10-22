# utils/model_loader.py
"""
Handles loading pretrained and custom models.
Designed to be modular for easy extension to new model sources or types.
"""

import timm
import torch.nn as nn
import torchvision.models as models

from config import DEVICE


def load_model(model_name: str) -> nn.Module:
    """
    Loads a specified model and prepares it for evaluation.

    This function can load models from torchvision, timm, or a local path.

    Args:
        model_name: The name of the model (e.g., 'resnet50') or path to a .pth file.

    Returns:
        A PyTorch model instance, in evaluation mode, on the configured device.

    Raises:
        ValueError: If the model_name is not recognized or the file path is invalid.
    """

    # # --- Future-proof logic for custom models ---
    # if model_name.endswith('.pth') or model_name.endswith('.pt'):
    #     try:
    #         # Here you would add logic to load your custom model architecture
    #         # and then load the state dict. For now, it's a placeholder.
    #         # Example:
    #         # model = MyCustomArch()
    #         # model.load_state_dict(torch.load(model_name, map_location=DEVICE))
    #         raise NotImplementedError("Custom model loading from path is not yet implemented.")
    #     except Exception as e:
    #         raise ValueError(f"Failed to load custom model from {model_name}: {e}")

    # --- Logic for loading from standard libraries ---
    try:
        # Try loading from torchvision first
        model = models.get_model(model_name, weights="IMAGENET1K_V1")
    except Exception:
        try:
            # If not in torchvision, try from timm
            model = timm.create_model(model_name, pretrained=True)
        except Exception:
            raise ValueError(
                f"Model '{model_name}' could not be found in torchvision or timm."
            )

    model.to(DEVICE)
    model.eval()

    return model
