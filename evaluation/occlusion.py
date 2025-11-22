"""
Occlusion-based evaluation utilities.

Handles progressive pixel occlusion and fill strategies for evaluating
attribution heatmaps by measuring model accuracy degradation.
"""

import numpy as np
import torch
from typing import Tuple, Dict, Callable
from functools import partial
from torchvision import transforms
from config import DEVICE


# -----------------
# Fill Strategy Implementations
# -----------------

# Cache blur transform to avoid repeated creation (performance optimization)
_BLUR_TRANSFORM_CACHE = None

# Cache random noise tensor to avoid repeated generation (performance optimization)
_RANDOM_NOISE_CACHE = None


def _get_blur_transform():
    """
    Get or create cached blur transform.
    
    Returns:
        GaussianBlur transform instance
    """
    global _BLUR_TRANSFORM_CACHE
    if _BLUR_TRANSFORM_CACHE is None:
        _BLUR_TRANSFORM_CACHE = transforms.GaussianBlur(kernel_size=21, sigma=10)
    return _BLUR_TRANSFORM_CACHE


def _get_random_noise(shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    """
    Get or create cached random noise tensor.
    
    Args:
        shape: Shape of the noise tensor
        device: Device to create tensor on
        
    Returns:
        Random noise tensor
    """
    global _RANDOM_NOISE_CACHE
    if (_RANDOM_NOISE_CACHE is None or 
        _RANDOM_NOISE_CACHE.shape != shape or 
        _RANDOM_NOISE_CACHE.device != device):
        _RANDOM_NOISE_CACHE = torch.rand(shape, device=device)
    return _RANDOM_NOISE_CACHE


def _fill_gray(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with solid gray color."""
    occluded_image = image.clone()
    occluded_image[:, mask] = 0.5
    return occluded_image


def _fill_blur(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with blurred version of image (optimized: reuse blurred result)."""
    blur_transform = _get_blur_transform()
    blurred_image = blur_transform(image)
    # Use blurred image as base and restore non-masked pixels (more efficient for large masks)
    if mask.sum() > mask.numel() * 0.5:  # If more than 50% masked
        occluded_image = blurred_image
        occluded_image[:, ~mask] = image[:, ~mask]
    else:
        occluded_image = image.clone()
        occluded_image[:, mask] = blurred_image[:, mask]
    return occluded_image


def _fill_random_noise(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with cached random noise."""
    occluded_image = image.clone()
    noise = _get_random_noise(image.shape, image.device)
    occluded_image[:, mask] = noise[:, mask]
    return occluded_image


def _fill_solid_color(image: torch.Tensor, mask: torch.Tensor, color) -> torch.Tensor:
    """
    Fill masked area with solid color.
    
    Args:
        image: Image tensor (C, H, W)
        mask: Boolean mask tensor (H, W)
        color: Color value(s) - can be single value or per-channel tuple
        
    Returns:
        Occluded image tensor
    """
    occluded_image = image.clone()
    
    # Handle per-channel colors (for normalized images)
    if isinstance(color, (tuple, list)):
        # Convert to tensor for vectorized assignment
        color_tensor = torch.tensor(color, dtype=image.dtype, device=image.device).view(3, 1)
        occluded_image[:, mask] = color_tensor
    else:
        # Single value for all channels
        occluded_image[:, mask] = color
    
    return occluded_image


def _fill_mean_color(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with mean color of the image (optimized: compute mean once)."""
    occluded_image = image.clone()
    # Compute mean per channel (more accurate than global mean)
    mean_colors = torch.mean(image, dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)
    occluded_image[:, mask] = mean_colors.expand_as(occluded_image)[:, mask]
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
        Flattened array of pixel indices sorted by attribution value (ascending)
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
        strategy: Fill strategy to use (must be in FILL_STRATEGY_REGISTRY)
        image_shape: (Height, Width) of the image
        
    Returns:
        Occluded image as new tensor
        
    Raises:
        ValueError: If strategy is not recognized or occlusion_level is invalid
    """
    if strategy not in FILL_STRATEGY_REGISTRY:
        raise ValueError(f"Fill strategy '{strategy}' is not recognized.")
    
    if not (0 <= occlusion_level <= 100):
        raise ValueError("Occlusion level must be between 0 and 100.")
    
    total_pixels = image_shape[0] * image_shape[1]
    num_pixels_to_occlude = int(total_pixels * (occlusion_level / 100.0))
    
    # Early return: no occlusion needed
    if num_pixels_to_occlude == 0:
        return image
    
    # Ensure image is on the correct device (GPU for performance)
    # Only transfer if not already on target device
    if image.device.type != DEVICE:
        image = image.to(DEVICE, non_blocking=True)
    
    # Select least important pixels to occlude
    pixels_to_occlude_flat = sorted_pixel_indices[:num_pixels_to_occlude]
    
    # Convert flat indices to 2D coordinates (vectorized numpy operation)
    rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)
    
    # Pre-allocate mask on GPU and fill efficiently
    mask = torch.zeros(image_shape, dtype=torch.bool, device=DEVICE)
    
    # Convert to torch tensors once (minimize conversions)
    rows_tensor = torch.from_numpy(rows).to(DEVICE, non_blocking=True)
    cols_tensor = torch.from_numpy(cols).to(DEVICE, non_blocking=True)
    mask[rows_tensor, cols_tensor] = True
    
    # Apply fill strategy
    fill_function = FILL_STRATEGY_REGISTRY[strategy]
    occluded_image = fill_function(image, mask)
    
    return occluded_image


def apply_occlusion_batch(
    images: list[torch.Tensor],
    sorted_pixel_indices_list: list[np.ndarray],
    occlusion_level: int,
    strategy: str,
    image_shape: Tuple[int, int] = (224, 224)
) -> list[torch.Tensor]:
    """
    Apply occlusion to a batch of images efficiently using vectorized operations.
    
    This function processes multiple images at once, which is much faster than
    calling apply_occlusion() in a loop, especially when images are already on GPU.
    
    Args:
        images: List of image tensors, each (C, H, W)
        sorted_pixel_indices_list: List of flattened arrays of pixel indices sorted by importance
        occlusion_level: Percentage (0-100) of pixels to occlude
        strategy: Fill strategy to use (must be in FILL_STRATEGY_REGISTRY)
        image_shape: (Height, Width) of the image
        
    Returns:
        List of occluded image tensors
        
    Raises:
        ValueError: If strategy is not recognized or occlusion_level is invalid
    """
    if strategy not in FILL_STRATEGY_REGISTRY:
        raise ValueError(f"Fill strategy '{strategy}' is not recognized.")
    
    if not (0 <= occlusion_level <= 100):
        raise ValueError("Occlusion level must be between 0 and 100.")
    
    if len(images) == 0:
        return []
    
    total_pixels = image_shape[0] * image_shape[1]
    num_pixels_to_occlude = int(total_pixels * (occlusion_level / 100.0))
    
    # Early return: no occlusion needed, return images directly (no cloning unless modified)
    if num_pixels_to_occlude == 0:
        return images  # No need to clone if not modifying
    
    # Ensure all images are on the same device (GPU for performance)
    device = DEVICE
    batch_size = len(images)
    
    # Check if images are already on target device to avoid unnecessary transfers
    if all(img.device.type == device for img in images):
        images_gpu = images  # No transfer needed
    else:
        images_gpu = [img.to(device, non_blocking=True) for img in images]
    
    # Pre-allocate all masks on GPU as single 3D tensor for better memory efficiency
    masks = torch.zeros((batch_size, image_shape[0], image_shape[1]), dtype=torch.bool, device=device)
    
    # Vectorized mask creation: process all images at once
    # Collect all row/col indices for vectorized assignment
    all_rows = []
    all_cols = []
    batch_indices = []
    
    for i, sorted_indices in enumerate(sorted_pixel_indices_list):
        if num_pixels_to_occlude > 0:
            pixels_to_occlude_flat = sorted_indices[:num_pixels_to_occlude]
            rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)
            
            all_rows.append(rows)
            all_cols.append(cols)
            batch_indices.append(np.full(len(rows), i, dtype=np.int64))
    
    # Concatenate all indices and convert to torch tensors once (minimize conversions)
    if all_rows:
        batch_idx_tensor = torch.from_numpy(np.concatenate(batch_indices)).to(device, non_blocking=True)
        rows_tensor = torch.from_numpy(np.concatenate(all_rows)).to(device, non_blocking=True)
        cols_tensor = torch.from_numpy(np.concatenate(all_cols)).to(device, non_blocking=True)
        
        # Vectorized mask assignment: set all pixels at once across all images
        masks[batch_idx_tensor, rows_tensor, cols_tensor] = True
    
    # Get fill function
    fill_function = FILL_STRATEGY_REGISTRY[strategy]
    
    # Apply occlusion to all images (still need per-image processing for fill strategies)
    # Note: Some fill strategies (like blur, mean) need per-image context
    occluded_images = []
    for i, image in enumerate(images_gpu):
        mask = masks[i]
        occluded_image = fill_function(image, mask)
        occluded_images.append(occluded_image)
    
    # Cleanup: masks tensor freed automatically when out of scope
    return occluded_images


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
