# modules/occlusion_evaluator.py
"""
Handles the core logic of Phase 2: progressive occlusion and evaluation.

This module takes an image and its corresponding heatmap, applies various
occlusion strategies, and evaluates the performance of independent "judging"
models on the resulting masked images.
"""
from typing import Tuple, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from config import DEVICE

from functools import partial


# --- Fill Strategy Implementations ---

def _fill_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fills the masked area with a solid gray color (0.5)."""
    # Create a copy to avoid modifying the original image
    occluded_image = image.clone()
    # Gray color has the same value for R, G, B channels
    occluded_image[:, mask] = 0.5
    return occluded_image


def _fill_blur(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fills the masked area with a blurred version of the image."""
    occluded_image = image.clone()

    # Create a blurred version of the entire image
    blur_transform = transforms.GaussianBlur(kernel_size=21, sigma=10)
    blurred_image = blur_transform(image)

    # Copy pixels from the blurred version only in the masked area
    occluded_image[:, mask] = blurred_image[:, mask]
    return occluded_image


def _fill_random_noise(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fills the masked area with random noise."""
    occluded_image = image.clone()

    # Generate random noise with the same shape as the image
    noise = torch.rand_like(image)

    # Copy pixels from the noise tensor only in the masked area
    occluded_image[:, mask] = noise[:, mask]
    return occluded_image


def _fill_solid_color(image: torch.Tensor, mask: torch.Tensor, color: float) -> torch.Tensor:
    """Fills the masked area with a solid color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = color
    return occluded_image


def _fill_mean_color(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fills the masked area with a mean color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = torch.mean(image[:, mask], dim=1, keepdim=True)
    return occluded_image


# --- Dispatcher for Fill Strategies ---
# To add a new strategy, implement a function like the ones above
# and add it to this dictionary.
FILL_STRATEGY_REGISTRY: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "gray": partial(_fill_solid_color, color=0.5),
    "black": partial(_fill_gray, color=0.0),
    "white": partial(_fill_gray, color=1.0),
    "blur": _fill_blur,
    "random_noise": _fill_random_noise,
    "mean": _fill_mean_color,
}


def sort_pixels(heatmap: np.ndarray) -> np.ndarray:
    """
    Sorts pixel indices from least to most important based on the heatmap.

    Args:
        heatmap: A 2D numpy array representing the attribution map.

    Returns:
        A flattened array of pixel indices, sorted by their attribution value.
    """
    # Flatten the heatmap and get the indices that would sort it in ascending order
    # This means the least important pixels come first.
    return np.argsort(heatmap.flatten())


def apply_occlusion(
        image: torch.Tensor,
        sorted_pixel_indices: np.ndarray,
        occlusion_level: int,
        strategy: str,
        image_shape: Tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """
    Applies occlusion to an image based on the sorted pixel importance.

    Args:
        image: The original image tensor (C, H, W).
        sorted_pixel_indices: A flattened array of pixel indices sorted by importance.
        occlusion_level: The percentage (0-100) of pixels to occlude.
        strategy: The fill strategy to use (e.g., 'gray', 'blur').
        image_shape: The (Height, Width) of the image.

    Returns:
        The occluded image as a new tensor.
    """
    if strategy not in FILL_STRATEGY_REGISTRY:
        raise ValueError(f"Fill strategy '{strategy}' is not recognized.")

    if not (0 <= occlusion_level <= 100):
        raise ValueError("Occlusion level must be between 0 and 100.")

    total_pixels = image_shape[0] * image_shape[1]
    num_pixels_to_occlude = int(total_pixels * (occlusion_level / 100.0))

    if num_pixels_to_occlude == 0:
        return image.clone()

    # Select the indices of the least important pixels to occlude
    pixels_to_occlude_flat = sorted_pixel_indices[:num_pixels_to_occlude]

    # Convert flat indices to 2D coordinates
    rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)

    # Create a boolean mask for the pixels to be occluded
    mask = torch.zeros(image_shape, dtype=torch.bool, device=DEVICE)
    mask[rows, cols] = True

    # Get the fill function from the registry and apply it
    fill_function = FILL_STRATEGY_REGISTRY[strategy]
    occluded_image = fill_function(image, mask)

    return occluded_image


def evaluate_judging_model(
        judging_model: nn.Module,
        masked_image: torch.Tensor,
        true_label: int
) -> int:
    """
    Evaluates a judging model's prediction on a masked image.

    Args:
        judging_model: The pre-trained model to use for evaluation.
        masked_image: The occluded image tensor (with batch dimension).
        true_label: The correct class index for the image.

    Returns:
        1 if the prediction is correct, 0 otherwise.
    """
    with torch.no_grad():
        output = judging_model(masked_image)
        # For models from `timm` or `transformers`, output might be a dict/tuple
        if isinstance(output, tuple):
            output = output[0]
        if isinstance(output, dict):
            output = output['logits']

        prediction = torch.argmax(output, dim=1).item()

    return 1 if prediction == true_label else 0
