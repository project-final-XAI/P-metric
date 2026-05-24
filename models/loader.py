"""
Model loading utilities.

Handles loading pretrained models and custom weight checkpoints.
Provides model provider interfaces and factory logic.
"""

from abc import ABC, abstractmethod
import logging
import os
import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE


# --- Shared Helper ---

def _prepare_model(model: nn.Module) -> nn.Module:
    """Prepare model for evaluation (eval mode, gradients enabled for CAM)."""
    model = model.to(DEVICE)
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception:
        pass

    model.eval()

    # Required for CAM methods to hook into gradients
    for param in model.parameters():
        param.requires_grad = True

    return model


# --- Model Provider Interfaces ---

class BaseModelProvider(ABC):
    """Defines the contract for all model providers."""

    @abstractmethod
    def get_model(self, model_name: str) -> nn.Module:
        """Load and return a ready-to-use PyTorch model."""
        pass


class PretrainedModelProvider(BaseModelProvider):
    """Loads pretrained models from torchvision (with timm fallback) for ImageNet."""

    def __init__(self) -> None:
        self._cache: dict[str, nn.Module] = {}

    def get_model(self, model_name: str) -> nn.Module:
        if model_name in self._cache:
            return self._cache[model_name]

        try:
            # Modern torchvision weights API
            model = models.get_model(model_name, weights="DEFAULT")
        except Exception:
            try:
                import timm
                model = timm.create_model(model_name, pretrained=True)
            except Exception as e:
                raise ValueError(f"Model '{model_name}' not found in torchvision or timm: {e}")

        model = _prepare_model(model)
        self._cache[model_name] = model
        logging.info(f"PretrainedModelProvider: loaded '{model_name}' on {DEVICE}")
        return model


class CustomModelProvider(BaseModelProvider):
    """Loads custom-trained models from local .pth weight files for SIPaKMeD."""

    def __init__(self, num_classes: int = 5) -> None:
        self._cache: dict[str, nn.Module] = {}
        self.num_classes = num_classes

    def get_model(self, model_name: str) -> nn.Module:
        if model_name in self._cache:
            return self._cache[model_name]

        # 1. Resolve local path
        weights_path = self._resolve_weights_path(model_name)

        # 2. Build the base architecture dynamically
        if model_name.startswith("efficientnet"):
            model = models.get_model(model_name.split("_sipakmed")[0], weights=None)
            in_features = model.classifier[1].in_features

            # If you used the same custom head for EfficientNet, it goes here!
            # Note: EfficientNet's classifier is named 'classifier'
            model.classifier = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(512, self.num_classes)
            )

        elif model_name.startswith("resnet"):
            model = models.get_model(model_name.split("_sipakmed")[0], weights=None)
            in_features = model.fc.in_features

            # Your exact custom head for ResNet!
            # Note: ResNet's classifier is named 'fc'
            model.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(512, self.num_classes)
            )
        else:
            raise ValueError(f"Unsupported custom architecture prefix in: {model_name}")

        # 3. Load the weights
        checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)

        # Handle dict vs raw state_dict checkpoints
        if isinstance(checkpoint, dict):
            state_key = 'state_dict' if 'state_dict' in checkpoint else 'model_state_dict'
            state_dict = checkpoint.get(state_key, checkpoint) # Fallback to dict itself
            model.load_state_dict(state_dict, strict=False)
        else:
            # Fallback if the user saved the entire model object (not recommended, but supported)
            model = checkpoint

        model = _prepare_model(model)
        self._cache[model_name] = model
        logging.info(f"CustomModelProvider: loaded '{model_name}' from {weights_path}")
        return model

    @staticmethod
    def _resolve_weights_path(model_name: str) -> str:
        """Find the .pth file dynamically."""
        # Note: Ensure your .pth files are actually stored in the directory this points to!
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Attempt 1: Exact match if user provided .pth extension
        if model_name.endswith('.pth') and os.path.exists(os.path.join(script_dir, model_name)):
            return os.path.join(script_dir, model_name)

        # Attempt 2: Auto-append _sipakmed.pth
        auto_path = os.path.join(script_dir, f"{model_name}_sipakmed.pth")
        if os.path.exists(auto_path):
            return auto_path

        # Attempt 3: Just .pth
        basic_path = os.path.join(script_dir, f"{model_name}.pth")
        if os.path.exists(basic_path):
            return basic_path

        raise FileNotFoundError(f"Could not find weights for {model_name} in {script_dir}")


# --- Factory ---

def get_model_provider(dataset_name: str) -> BaseModelProvider:
    """Factory: return the appropriate model provider for the targeted dataset."""
    if dataset_name == "imagenet":
        return PretrainedModelProvider()
    elif dataset_name in ("SIPaKMeD", "SIPaKMeD_cropped"):
        return CustomModelProvider(num_classes=5)
    else:
        raise ValueError(f"Unknown dataset for model provision: {dataset_name}")