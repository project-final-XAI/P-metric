"""
P-Metric implementation for comparison with SSMS.

P-Metric evaluates faithfulness through progressive occlusion at multiple levels.
This is the expensive multi-step evaluation that SSMS aims to replace.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from typing import List, Tuple, Dict

# Import existing utilities
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import calculate_auc, calculate_drop
from evaluation.occlusion import sort_pixels, apply_occlusion


def compute_pmetric(
    heatmap: np.ndarray,
    image: torch.Tensor,
    judge_model,
    true_label: int,
    occlusion_percents: List[int],
    fill_strategy: str = 'mean'
) -> Tuple[Dict[str, float], List[float]]:
    """
    Compute P-Metric (progressive occlusion-based faithfulness metric).
    
    For each occlusion level P:
    1. Sort pixels by heatmap importance (least important first)
    2. Occlude bottom P% of pixels using fill_strategy
    3. Evaluate judge model on occluded image
    4. Calculate accuracy Acc(P)
    
    Then compute:
    - AUC: Area under Acc(P) curve
    - DROP: Acc(0) - Acc(75)
    - InflectionPoint_max_slope: P with maximum negative slope
    - InflectionPoint_threshold: First P where Acc < Acc(0) - 0.1
    
    Args:
        heatmap: 2D numpy array representing attribution map (H, W)
        image: Normalized image tensor (C, H, W) - already preprocessed
        judge_model: Judge model with predict() method
        true_label: True class label (integer)
        occlusion_percents: List of occlusion percentages to evaluate [0, 5, 10, ...]
        fill_strategy: Fill strategy for occlusion ('mean', 'gray', 'black', etc.)
        
    Returns:
        Tuple of (metrics_dict, accuracy_curve)
        - metrics_dict: dict with keys: AUC, DROP, InflectionPoint_max_slope, InflectionPoint_threshold
        - accuracy_curve: List of Acc(P) values for each occlusion level
    """
    # Ensure heatmap is 2D numpy array
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().detach().numpy()
    
    # Clip heatmap to non-negative
    heatmap = np.maximum(heatmap, 0)
    
    # Sort pixels by importance (ascending: least important first)
    sorted_pixel_indices = sort_pixels(heatmap)
    
    # Get image shape
    if image.ndim == 3:
        _, h, w = image.shape
    else:
        h, w = 224, 224  # Default ImageNet size
    
    # Evaluate at each occlusion level
    accuracies = []
    
    # Ensure image is on the correct device (same as judge model)
    # Get device from judge model if available
    device = image.device
    if hasattr(judge_model, 'device'):
        device = judge_model.device
    elif hasattr(judge_model, 'model'):
        try:
            device = next(judge_model.model.parameters()).device
        except (StopIteration, AttributeError):
            pass
    
    image = image.to(device)
    
    for occlusion_level in occlusion_percents:
        # Apply occlusion
        occluded_image = apply_occlusion(
            image,
            sorted_pixel_indices,
            occlusion_level,
            fill_strategy,
            image_shape=(h, w)
        )
        
        # Ensure occluded image is on correct device (apply_occlusion might use DEVICE from config)
        occluded_image = occluded_image.to(device)
        
        # Get prediction from judge model
        occluded_batch = occluded_image.unsqueeze(0)  # Add batch dimension
        predictions = judge_model.predict(occluded_batch)
        predicted_label = int(predictions[0])
        
        # Calculate accuracy (binary: 1 if correct, 0 if wrong)
        accuracy = 1.0 if predicted_label == true_label else 0.0
        accuracies.append(accuracy)
    
    # Compute metrics
    baseline_accuracy = accuracies[0] if accuracies else 0.0
    
    # AUC: area under accuracy curve
    auc = calculate_auc(accuracies, occlusion_percents)
    
    # DROP: accuracy drop at 75% occlusion
    drop = calculate_drop(accuracies, occlusion_percents, baseline_accuracy, drop_level=75)
    if np.isnan(drop):
        drop = 0.0
    
    # InflectionPoint_max_slope: find P with maximum negative slope
    inflection_max_slope = _find_max_slope_inflection(accuracies, occlusion_percents)
    
    # InflectionPoint_threshold: first P where Acc < Acc(0) - 0.1
    inflection_threshold = _find_threshold_inflection(accuracies, occlusion_percents, baseline_accuracy)
    
    metrics = {
        'AUC': float(auc),
        'DROP': float(drop),
        'InflectionPoint_max_slope': float(inflection_max_slope),
        'InflectionPoint_threshold': float(inflection_threshold)
    }
    
    return metrics, accuracies


def _find_max_slope_inflection(accuracies: List[float], occlusion_percents: List[int]) -> float:
    """
    Find occlusion level with maximum negative slope (steepest drop).
    
    Args:
        accuracies: List of accuracy values
        occlusion_percents: List of occlusion percentages
        
    Returns:
        Occlusion percentage with maximum negative slope
    """
    if len(accuracies) < 2:
        return 0.0
    
    # Calculate slopes between consecutive points
    slopes = []
    for i in range(len(accuracies) - 1):
        delta_acc = accuracies[i + 1] - accuracies[i]
        delta_p = occlusion_percents[i + 1] - occlusion_percents[i]
        if delta_p > 0:
            slope = delta_acc / delta_p
            slopes.append((occlusion_percents[i], slope))
    
    if not slopes:
        return 0.0
    
    # Find point with most negative slope
    min_slope_idx = min(range(len(slopes)), key=lambda i: slopes[i][1])
    return slopes[min_slope_idx][0]


def _find_threshold_inflection(
    accuracies: List[float],
    occlusion_percents: List[int],
    baseline_accuracy: float,
    threshold: float = 0.1
) -> float:
    """
    Find first occlusion level where accuracy drops below baseline - threshold.
    
    Args:
        accuracies: List of accuracy values
        occlusion_percents: List of occlusion percentages
        baseline_accuracy: Accuracy at 0% occlusion
        threshold: Drop threshold (default 0.1)
        
    Returns:
        First occlusion percentage where Acc < baseline - threshold, or last level if never reached
    """
    target_accuracy = baseline_accuracy - threshold
    
    for i, acc in enumerate(accuracies):
        if acc < target_accuracy:
            return float(occlusion_percents[i])
    
    # If never reached, return last occlusion level
    return float(occlusion_percents[-1]) if occlusion_percents else 0.0


