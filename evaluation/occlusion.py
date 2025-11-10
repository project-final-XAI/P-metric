"""
Occlusion-based evaluation utilities.

Handles progressive pixel occlusion and fill strategies.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Callable
from functools import partial
from torchvision import transforms
from config import DEVICE


# Fill Strategy Implementations

# Cache blur transform to avoid repeated creation (performance optimization)
_BLUR_TRANSFORM_CACHE = None
# Cache random noise tensor to avoid repeated generation (much faster!)
_RANDOM_NOISE_CACHE = None

def _get_blur_transform():
    """Get or create cached blur transform."""
    global _BLUR_TRANSFORM_CACHE
    if _BLUR_TRANSFORM_CACHE is None:
        _BLUR_TRANSFORM_CACHE = transforms.GaussianBlur(kernel_size=21, sigma=10)
    return _BLUR_TRANSFORM_CACHE

def _get_random_noise(shape, device):
    """Get or create cached random noise tensor."""
    global _RANDOM_NOISE_CACHE
    if _RANDOM_NOISE_CACHE is None or _RANDOM_NOISE_CACHE.shape != shape or _RANDOM_NOISE_CACHE.device != device:
        _RANDOM_NOISE_CACHE = torch.rand(shape, device=device)
    return _RANDOM_NOISE_CACHE

def _fill_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with solid gray color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = 0.5
    return occluded_image


def _fill_blur(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with blurred version of image."""
    occluded_image = image.clone()
    blur_transform = _get_blur_transform()  # Use cached transform
    blurred_image = blur_transform(image)
    occluded_image[:, mask] = blurred_image[:, mask]
    return occluded_image


def _fill_random_noise(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with cached random noise (faster than generating each time)."""
    occluded_image = image.clone()
    noise = _get_random_noise(image.shape, image.device)
    occluded_image[:, mask] = noise[:, mask]
    return occluded_image


def _fill_solid_color(image: torch.Tensor, mask: torch.Tensor, color) -> torch.Tensor:
    """Fill masked area with solid color (can be single value or per-channel tuple)."""
    occluded_image = image.clone()
    
    # Handle per-channel colors (for normalized images)
    if isinstance(color, (tuple, list)):
        # color is (R, G, B) values for each channel
        for c in range(3):
            occluded_image[c, mask] = color[c]
    else:
        # Single value for all channels
        occluded_image[:, mask] = color
    
    return occluded_image


def _fill_mean_color(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with mean color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = torch.mean(image[:, mask], dim=1, keepdim=True)
    return occluded_image


# ImageNet normalization values: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
# Properly normalized colors for black/white (to show correctly after denormalization):
# Black (0,0,0): (0 - mean) / std
NORMALIZED_BLACK = (-2.118, -2.036, -1.804)
# White (1,1,1): (1 - mean) / std  
NORMALIZED_WHITE = (2.249, 2.429, 2.640)

# Fill Strategy Registry
FILL_STRATEGY_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "gray": partial(_fill_solid_color, color=0.5),
    "black": partial(_fill_solid_color, color=NORMALIZED_BLACK),
    "white": partial(_fill_solid_color, color=NORMALIZED_WHITE),
    "blur": _fill_blur,
    "random_noise": _fill_random_noise,
    "mean": _fill_mean_color,
}


def sort_pixels(heatmap: np.ndarray) -> np.ndarray:
    """
    Sort pixel indices from least to most important based on heatmap.
    
    Args:
        heatmap: 2D numpy array representing attribution map
        
    Returns:
        Flattened array of pixel indices sorted by attribution value
    """
    return np.argsort(heatmap.flatten())


def apply_occlusion(
        image: torch.Tensor,
        sorted_pixel_indices: np.ndarray,
        occlusion_level: int,
        strategy: str,
        image_shape: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Apply occlusion to image based on sorted pixel importance.
    
    Args:
        image: Original image tensor (C, H, W)
        sorted_pixel_indices: Flattened array of pixel indices sorted by importance
        occlusion_level: Percentage (0-100) of pixels to occlude
        strategy: Fill strategy to use
        image_shape: (Height, Width) of the image
        
    Returns:
        Occluded image as new tensor
    """
    if strategy not in FILL_STRATEGY_REGISTRY:
        raise ValueError(f"Fill strategy '{strategy}' is not recognized.")

    if not (0 <= occlusion_level <= 100):
        raise ValueError("Occlusion level must be between 0 and 100.")

    total_pixels = image_shape[0] * image_shape[1]
    num_pixels_to_occlude = int(total_pixels * (occlusion_level / 100.0))

    if num_pixels_to_occlude == 0:
        return image.clone()

    # Move the image to the same device as the torch (GPU obviously..), by the way - that BUG cost me something like 45 minute so a lot of respect to him
    image = image.to(DEVICE)

    # Select least important pixels to occlude
    pixels_to_occlude_flat = sorted_pixel_indices[:num_pixels_to_occlude]

    # Convert flat indices to 2D coordinates (more efficient with advanced indexing)
    rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)

    # Pre-allocate mask on GPU and fill efficiently
    mask = torch.zeros(image_shape, dtype=torch.bool, device=DEVICE)
    
    # Use torch tensors for indexing (faster than numpy indexing)
    rows_tensor = torch.from_numpy(rows).to(DEVICE)
    cols_tensor = torch.from_numpy(cols).to(DEVICE)
    mask[rows_tensor, cols_tensor] = True

    # Apply fill strategy
    fill_function = FILL_STRATEGY_REGISTRY[strategy]
    occluded_image = fill_function(image, mask)

    return occluded_image


def evaluate_judging_model(
        judging_model: torch.nn.Module,
        masked_image: torch.Tensor,
        true_label: int
) -> int:
    """
    Evaluate judging model's prediction on masked image.
    
    Args:
        judging_model: Pre-trained model for evaluation
        masked_image: Occluded image tensor (with batch dimension)
        true_label: Correct class index for the image
        
    Returns:
        1 if prediction is correct, 0 otherwise
    """
    with torch.no_grad():
        output = judging_model(masked_image)

        # Handle different output formats
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output['logits']

        prediction = torch.argmax(output, dim=1).item()

    return 1 if prediction == true_label else 0
