"""
Quantitative faithfulness metrics calculation.

Calculates AUC and DROP metrics from accuracy-degradation curves.
"""

from typing import List
import numpy as np


def calculate_auc(accuracy_points: List[float], occlusion_levels: List[int]) -> float:
    """
    Calculate Area Under Curve for accuracy-degradation curve.
    
    Args:
        accuracy_points: List of accuracy values for each occlusion level
        occlusion_levels: List of occlusion percentages
        
    Returns:
        AUC score normalized to [0, 1]
    """
    # Sort by occlusion level for correct integration
    sorted_pairs = sorted(zip(occlusion_levels, accuracy_points))
    sorted_levels = [p[0] for p in sorted_pairs]
    sorted_accuracies = [p[1] for p in sorted_pairs]

    # Use trapezoidal rule for numerical integration
    # Normalize by total range (100)
    return np.trapz(sorted_accuracies, x=sorted_levels) / 100.0


def calculate_drop(
        accuracy_points: List[float],
        occlusion_levels: List[int],
        initial_accuracy: float,
        drop_level: int = 75
) -> float:
    """
    Calculate DROP-in accuracy at specific occlusion level.
    
    Args:
        accuracy_points: List of accuracy values
        occlusion_levels: List of occlusion percentages
        initial_accuracy: Model accuracy with 0% occlusion
        drop_level: Occlusion percentage to measure drop at
        
    Returns:
        Accuracy drop value, NaN if level not found
    """
    try:
        idx = occlusion_levels.index(drop_level)
        accuracy_at_level = accuracy_points[idx]
        return initial_accuracy - accuracy_at_level
    except ValueError:
        return float('nan')

