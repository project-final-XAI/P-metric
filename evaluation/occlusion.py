"""
Occlusion-based evaluation utilities.

Handles progressive pixel occlusion and fill strategies.
"""

import numpy as np
import torch
import logging
from typing import Tuple, Dict, Callable
from functools import partial
from torchvision import transforms
from config import DEVICE


# Fill Strategy Implementations

def _fill_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with solid gray color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = 0.5
    return occluded_image


def _fill_blur(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with blurred version of image."""
    occluded_image = image.clone()
    blur_transform = transforms.GaussianBlur(kernel_size=21, sigma=10)
    blurred_image = blur_transform(image)
    occluded_image[:, mask] = blurred_image[:, mask]
    return occluded_image


def _fill_random_noise(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with random noise."""
    occluded_image = image.clone()
    noise = torch.rand_like(image)
    occluded_image[:, mask] = noise[:, mask]
    return occluded_image


def _fill_solid_color(image: torch.Tensor, mask: torch.Tensor, color: float) -> torch.Tensor:
    """Fill masked area with solid color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = color
    return occluded_image


def _fill_mean_color(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with mean color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = torch.mean(image[:, mask], dim=1, keepdim=True)
    return occluded_image


# Fill Strategy Registry
FILL_STRATEGY_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "gray": partial(_fill_solid_color, color=0.5),
    "black": partial(_fill_solid_color, color=0.0),
    "white": partial(_fill_solid_color, color=1.0),
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

    # Select least important pixels to occlude
    pixels_to_occlude_flat = sorted_pixel_indices[:num_pixels_to_occlude]

    # Convert flat indices to 2D coordinates
    rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)

    # Create boolean mask for pixels to be occluded
    mask = torch.zeros(image_shape, dtype=torch.bool, device=DEVICE)
    mask[rows, cols] = True

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

