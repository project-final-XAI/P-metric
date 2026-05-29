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
def _fill_blur(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Fill masked area with blurred version of image."""
    blur_transform = transforms.GaussianBlur(kernel_size=21, sigma=10)
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
    """Fill masked area with random noise."""
    occluded_image = image.clone()
    noise = torch.rand(image.shape, device=image.device)
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
    mean_colors = torch.mean(image, dim=(1, 2))  # Shape: (C,)
    # Use same approach as _fill_solid_color - reshape to (C, 1) for broadcasting
    mean_colors = mean_colors.unsqueeze(1)  # Shape: (C, 1)
    occluded_image[:, mask] = mean_colors
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
    Sort pixel indices from least to most important based on heatmap magnitude.
    
    Args:
        heatmap: 2D numpy array representing attribution map
        
    Returns:
        Flattened array of pixel indices sorted by attribution magnitude (ascending)
    """
    return np.argsort(np.abs(heatmap.flatten()))


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
    
    # Early return: no occlusion needed, but clone to avoid reference issues
    if num_pixels_to_occlude == 0:
        return image.clone()
    
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


def build_occlusion_mask_batch(
    sorted_pixel_indices_list: list[np.ndarray],
    occlusion_level: int,
    image_shape: Tuple[int, int] = (224, 224),
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Build a single (B, H, W) bool mask shared by all fill strategies.

    Mask construction is independent of the fill strategy, so callers can
    compute it once per (model, method, level) and reuse across the 6 fills.
    """
    if not (0 <= occlusion_level <= 100):
        raise ValueError("Occlusion level must be between 0 and 100.")

    if device is None:
        device = DEVICE

    batch_size = len(sorted_pixel_indices_list)
    masks = torch.zeros(
        (batch_size, image_shape[0], image_shape[1]),
        dtype=torch.bool,
        device=device,
    )
    if batch_size == 0:
        return masks

    total_pixels = image_shape[0] * image_shape[1]
    num_pixels_to_occlude = int(total_pixels * (occlusion_level / 100.0))
    if num_pixels_to_occlude == 0:
        return masks

    all_rows: list[np.ndarray] = []
    all_cols: list[np.ndarray] = []
    batch_indices: list[np.ndarray] = []

    for i, sorted_indices in enumerate(sorted_pixel_indices_list):
        pixels_to_occlude_flat = sorted_indices[:num_pixels_to_occlude]
        rows, cols = np.unravel_index(pixels_to_occlude_flat, image_shape)
        all_rows.append(rows)
        all_cols.append(cols)
        batch_indices.append(np.full(len(rows), i, dtype=np.int64))

    batch_idx_tensor = torch.from_numpy(np.concatenate(batch_indices)).to(
        device, non_blocking=True
    )
    rows_tensor = torch.from_numpy(np.concatenate(all_rows)).to(device, non_blocking=True)
    cols_tensor = torch.from_numpy(np.concatenate(all_cols)).to(device, non_blocking=True)
    masks[batch_idx_tensor, rows_tensor, cols_tensor] = True
    return masks


def apply_fill_to_batch(
    batch: torch.Tensor,
    masks: torch.Tensor,
    strategy: str,
) -> torch.Tensor:
    """Apply a fill strategy to a stacked (B, C, H, W) batch in one shot.

    For ``gray/black/white/random_noise/mean`` we use ``torch.where`` so the
    whole batch is processed in a single vectorized operation. ``blur`` still
    iterates per image because its kernel depends on the image content.
    """
    if strategy not in FILL_STRATEGY_REGISTRY:
        raise ValueError(f"Fill strategy '{strategy}' is not recognized.")

    if batch.ndim != 4:
        raise ValueError(f"Expected (B, C, H, W) batch, got shape {tuple(batch.shape)}")
    if masks.ndim != 3 or masks.shape[0] != batch.shape[0]:
        raise ValueError(
            f"Mask shape {tuple(masks.shape)} incompatible with batch {tuple(batch.shape)}"
        )

    # Broadcast (B, H, W) -> (B, 1, H, W) for channelwise mixing.
    mask_b1hw = masks.unsqueeze(1)

    if strategy == "gray":
        return torch.where(mask_b1hw, batch.new_full((), 0.5), batch)

    if strategy == "black":
        color = torch.tensor(NORMALIZED_BLACK, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
        return torch.where(mask_b1hw, color, batch)

    if strategy == "white":
        color = torch.tensor(NORMALIZED_WHITE, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
        return torch.where(mask_b1hw, color, batch)

    if strategy == "random_noise":
        noise = torch.rand_like(batch)
        return torch.where(mask_b1hw, noise, batch)

    if strategy == "mean":
        # Per-image, per-channel mean broadcast to (B, C, 1, 1).
        mean_colors = batch.mean(dim=(2, 3), keepdim=True)
        return torch.where(mask_b1hw, mean_colors.expand_as(batch), batch)

    if strategy == "blur":
        # Blur kernel is fixed but must run on each image; we keep the loop for
        # this strategy only.  The rest of Phase 2 still benefits from the
        # batched mask + stacked tensor input.
        out = batch.clone()
        fill_function = FILL_STRATEGY_REGISTRY["blur"]
        for i in range(batch.shape[0]):
            out[i] = fill_function(batch[i], masks[i])
        return out

    # Fallback: unknown registered strategy - go through the per-image API.
    fill_function = FILL_STRATEGY_REGISTRY[strategy]
    out = batch.clone()
    for i in range(batch.shape[0]):
        out[i] = fill_function(batch[i], masks[i])
    return out

