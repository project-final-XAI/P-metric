"""
Single-Step Masking Score (SSMS) implementation.

SSMS evaluates faithfulness by applying a single adaptive mask to the image
and checking if the judge model still predicts correctly. This is much faster
than P-Metric which requires multiple occlusion levels.
"""

import torch
import numpy as np
from typing import Tuple


def compute_ssms(
    heatmap: np.ndarray,
    image: torch.Tensor,
    judge_model,
    true_label: int,
    alpha_max: float = 10.0,
    eps: float = 1e-8,
    power_factor: float = 2.5,  # Higher power = stronger masking
    sparsity_penalty_factor: float = 3.0,  # Penalty for non-informative heatmaps
    base_alpha: float = 1.0,  # Base alpha multiplier
    return_masked_image: bool = False  # If True, also return masked_image for visualization
) -> Tuple[float, dict, torch.Tensor]:
    """
    Compute Single-Step Masking Score (SSMS) with optimal ratio-preserving masking.
    
    OPTIMAL MASKING FORMULA (preserves heatmap ratios perfectly):
    1. Normalize heatmap H to [0, 1]: H_norm = H / H_max
    2. Linear mapping: mask_base = H_norm (preserves ratios: if H1/H2 = r, then mask1/mask2 = r)
    3. Apply floor for sparse heatmaps: mask = mask_base + min_floor * exp(-k * mask_base)
       - Floor only affects zero values (preserves ratios for non-zero values)
       - Floor value depends on sparsity (sparse heatmaps get higher floor)
    4. Clip to [0, 1]: mask = clip(mask, 0, 1)
    5. Apply mask: I* = I * M[...,None]
    6. Get prediction on I* from judge_model
    7. SSMS_score = 1 if correct, 0 if wrong
    
    Key advantages:
    - Perfect ratio preservation: H1/H2 = mask1/mask2 (exactly!)
    - Continuous function (no thresholds, no piecewise parts)
    - Handles sparse heatmaps (minimum masking for unimportant pixels)
    - Simple and efficient (4 steps)
    
    Args:
        heatmap: 2D numpy array representing attribution map (H, W)
        image: Normalized image tensor (C, H, W) - already preprocessed
        judge_model: Judge model with predict() method (returns class indices)
        true_label: True class label (integer)
        alpha_max: (Deprecated - not used in new formula)
        eps: Small epsilon for numerical stability
        power_factor: (Deprecated - not used in new formula, uses linear instead)
        sparsity_penalty_factor: (Deprecated - not used in new formula)
        base_alpha: (Deprecated - not used in new formula)
        
    Returns:
        Tuple of (SSMS_score, metadata_dict, masked_image)
        - SSMS_score: Binary score (1.0 if correct prediction, 0.0 if wrong)
        - metadata: dict with keys: S, alpha, entropy, sparsity, penalty
        - masked_image: torch.Tensor of masked image (C, H, W) for visualization
    """
    # Ensure heatmap is 2D numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().detach().numpy()
    
    # Clip heatmap to non-negative values
    heatmap = np.maximum(heatmap, 0)
    
    # Normalize heatmap to [0, 1] if needed
    hmin, hmax = heatmap.min(), heatmap.max()
    if hmax > hmin:
        heatmap = (heatmap - hmin) / (hmax - hmin)
    else:
        # If all values are the same, return uniform mask (all zeros = full occlusion)
        mask = np.zeros_like(heatmap)
        device = image.device
        mask_tensor = torch.from_numpy(mask).float().to(device)
        
        # Apply mask to denormalized image (same as main logic)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        image_denorm = image * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        masked_image_denorm = image_denorm * mask_tensor[None, :, :]
        masked_image = (masked_image_denorm - mean) / std
        
        masked_batch = masked_image.unsqueeze(0)
        with torch.inference_mode():
            predictions = judge_model.predict(masked_batch)
        predicted_label = int(predictions[0])
        ssms_score = 1.0 if predicted_label == true_label else 0.0
        return ssms_score, {'S': float(heatmap.sum()), 'entropy': 0.0, 'sparsity': 1.0, 'alpha': 0.0, 'penalty': 0.0, 'power_factor': 1.0}, masked_image
    
    # Compute statistics
    S = heatmap.sum()
    N = heatmap.size
    
    # Calculate sparsity ratio (used for floor calculation)
    sparsity_ratio = S / N  # Ratio: 0 (sparse) to 1 (all ones)
    
    # OPTIMAL MASK FORMULA - Perfect ratio preservation + sparse heatmap handling
    # This formula preserves the exact ratios from the heatmap while handling sparse heatmaps
    
    # ============================================================================
    # STEP 1: Normalize heatmap to [0, 1]
    # ============================================================================
    # Why: Ensures all heatmap values are in the same scale [0, 1]
    # What: H_norm = H / H_max (if H_max > 0)
    # Result: If H1/H2 = r, then H_norm1/H_norm2 = r (ratios preserved!)
    # Example: H = [0.8, 0.6, 0.4] → H_norm = [1.0, 0.75, 0.5] (ratios: 0.75, 0.5 preserved)
    H_max = heatmap.max()
    if H_max > eps:
        H_norm = heatmap / H_max
    else:
        H_norm = heatmap
    
    # ============================================================================
    # STEP 2: Linear function for perfect ratio preservation
    # ============================================================================
    # Why: Linear function preserves ratios perfectly
    # What: mask_base = H_norm (linear, no transformation)
    # Result: If H1/H2 = r, then mask1/mask2 = r (exactly!)
    # Example: H_norm = [1.0, 0.8, 0.5] → mask_base = [1.0, 0.8, 0.5]
    #          Ratios: 0.8/1.0 = 0.8 ✓, 0.5/1.0 = 0.5 ✓ (perfect!)
    # Why not use power? Power would change ratios: 0.8^1.1/1.0^1.1 = 0.798 (not 0.8!)
    mask_base = H_norm
    
    # ============================================================================
    # STEP 3: Floor using exponential decay (only affects zero values)
    # ============================================================================
    # Problem: When heatmap is very sparse (only small part is important, rest is 0),
    #          we want unimportant pixels to get minimum masking (e.g., 20%) not 0%
    # Why: This creates more uniform distribution and prevents extreme sparsity
    # What: Add minimum floor value to zero/unimportant pixels
    
    # 3a. Calculate sparsity ratio (how sparse the heatmap is)
    #     OLD: sparsity_ratio = S/N (problem: doesn't distinguish between "few high values" vs "many low values")
    #     NEW: Use fraction of pixels with high importance (after normalization)
    #     This better captures "only small region important" vs "many regions slightly important"
    #     Example: H = [1.0, 0.01, 0.01, ...] → S/N might be low, but many pixels have low values
    #              We want to detect: only few pixels are truly important (> threshold)
    
    # Count pixels with high importance (e.g., > 0.1 of max after normalization)
    # If only small fraction has high importance, it's truly sparse
    high_importance_threshold = 0.1  # Pixels with > 10% of max value are "important"
    high_importance_count = np.sum(H_norm > high_importance_threshold)
    high_importance_ratio = high_importance_count / N  # Fraction of pixels that are "important"
    
    # Use this ratio for sparsity detection
    # low high_importance_ratio = truly sparse (only few pixels important) → need floor
    # high high_importance_ratio = not sparse (many pixels important) → no floor needed
    # Invert for sigmoid: we want floor when high_importance_ratio is LOW
    sparsity_ratio = 1.0 - high_importance_ratio  # Invert: low high_importance_ratio → high sparsity_ratio → floor
    
    # 3b. Calculate floor value using sigmoid (continuous, no thresholds)
    #     floor_sigmoid = 1 when sparse (low high_importance_ratio), 0 when not sparse (high ratio)
    #     This is a smooth transition based on sparsity
    #     Example: high_importance_ratio = 0.01 (1% important) → sparsity_ratio = 0.99 → floor_sigmoid ≈ 1.0
    #              high_importance_ratio = 0.5 (50% important) → sparsity_ratio = 0.5 → floor_sigmoid ≈ 0.0
    #     Threshold adjusted: 0.85 means floor applies when < 15% of pixels are important
    floor_sigmoid = 1.0 / (1.0 + np.exp(25.0 * (sparsity_ratio - 0.85)))  # Adjusted threshold for new metric
    # min_floor controls minimum masking for zero values
    # For very sparse heatmaps (only small region important), apply floor to prevent extreme blackout
    # For non-sparse heatmaps (many regions important), no floor (zero values become black)
    # This ensures sparse heatmaps don't cause extreme blackout while preserving black for non-sparse heatmaps
    min_floor = 0.15 * floor_sigmoid  # 0.15 for very sparse, 0.0 for non-sparse
    
    # 3c. Apply floor using exponential decay (only affects zero values)
    #     floor_transition = exp(-k * mask_base)
    #     - When mask_base = 0: exp(0) = 1.0 → full floor applied ✓
    #     - When mask_base = 0.1: exp(-10) ≈ 0.0 → no floor ✓
    #     - When mask_base = 1.0: exp(-100) ≈ 0.0 → no floor ✓
    #     This ensures floor ONLY affects zero/unimportant pixels, preserving ratios!
    #     Example: mask_base = [1.0, 0.8, 0.5, 0.1, 0.0]
    #              floor_transition = [0.0, 0.0, 0.0, 0.0, 1.0]
    #              mask = [1.0, 0.8, 0.5, 0.1, 0.15] (ratios preserved!)
    k = 100.0  # Decay rate (higher = faster decay, only affects values very close to 0)
    floor_transition = np.exp(-k * mask_base)
    mask = mask_base + min_floor * floor_transition
    
    # ============================================================================
    # STEP 4: Clip to [0, 1]
    # ============================================================================
    # Why: Ensure mask values are in valid range [0, 1]
    # What: mask = clip(mask, 0, 1)
    # Result: All mask values between 0 and 1
    mask = np.clip(mask, 0, 1)
    
    # Convert mask to tensor and ensure same device as image
    device = image.device
    # Use non_blocking transfer for better performance
    mask_tensor = torch.from_numpy(mask).float().to(device, non_blocking=True)
    
    # IMPORTANT: Apply mask to DENORMALIZED image, then re-normalize
    # This ensures that black pixels (0,0,0) stay black after masking
    # If we mask the normalized image directly, black becomes gray after denormalization!
    
    # Denormalize image (reverse ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
    image_denorm = image * std + mean
    image_denorm = torch.clamp(image_denorm, 0, 1)  # Ensure valid range [0, 1]
    
    # Apply mask to denormalized image: I*_denorm = I_denorm * M[...,None]
    # Broadcast mask from (H, W) to (C, H, W)
    masked_image_denorm = image_denorm * mask_tensor[None, :, :]
    
    # Re-normalize for judge model (judge expects normalized input)
    masked_image = (masked_image_denorm - mean) / std
    
    # Get prediction from judge model
    # Judge model expects batch dimension, so add it
    masked_batch = masked_image.unsqueeze(0)  # (1, C, H, W)
    # Use inference_mode for better performance (faster than no_grad)
    with torch.inference_mode():
        predictions = judge_model.predict(masked_batch)
    predicted_label = int(predictions[0])
    
    # SSMS score: binary (1 if correct, 0 if wrong)
    ssms_score = 1.0 if predicted_label == true_label else 0.0
    
    # Compute additional metadata for analysis
    # Entropy: measure of information content
    heatmap_flat = heatmap.flatten()
    # Normalize to probabilities
    prob = heatmap_flat / (heatmap_flat.sum() + eps)
    entropy = -np.sum(prob * np.log(prob + eps))
    
    # Sparsity: fraction of near-zero values
    threshold = 0.1  # Consider values < 10% of max as sparse
    sparsity = np.mean(heatmap < threshold)
    
    metadata = {
        'S': float(S),
        'alpha': 0.0,  # Not used in new formula (kept for compatibility)
        'entropy': float(entropy),
        'sparsity': float(sparsity),
        'penalty': 0.0,  # Not used in new formula (kept for compatibility)
        'power_factor': 1.0  # Linear (kept for compatibility)
    }
    
    return ssms_score, metadata, masked_image


