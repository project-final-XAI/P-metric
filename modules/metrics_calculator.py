# modules/metrics_calculator.py
"""
Calculates quantitative faithfulness metrics from accuracy-degradation curves.
"""
from typing import List

import numpy as np


def calculate_auc(accuracy_points: List[float], occlusion_levels: List[int]) -> float:
    """
    Calculates the Area Under the Curve (AUC) for an accuracy-degradation curve.

    Args:
        accuracy_points: A list of accuracy values corresponding to each occlusion level.
        occlusion_levels: A list of the occlusion percentages.

    Returns:
        The AUC score, normalized to be between 0 and 1.
    """
    # Ensure the points are sorted by occlusion level for correct integration
    sorted_pairs = sorted(zip(occlusion_levels, accuracy_points))
    sorted_levels = [p[0] for p in sorted_pairs]
    sorted_accuracies = [p[1] for p in sorted_pairs]

    # Use numpy's trapezoidal rule for numerical integration
    # We normalize by the total range of occlusion levels (100)
    return np.trapz(sorted_accuracies, x=sorted_levels) / 100.0


def calculate_drop(
        accuracy_points: List[float],
        occlusion_levels: List[int],
        initial_accuracy: float,
        drop_level: int = 75
) -> float:
    """
    Calculates the DROP-in accuracy at a specific occlusion level.
    The paper defines this at P=75%[cite: 185].

    Args:
        accuracy_points: A list of accuracy values.
        occlusion_levels: A list of the occlusion percentages.
        initial_accuracy: The model's accuracy with 0% occlusion.
        drop_level: The occlusion percentage at which to measure the drop.

    Returns:
        The accuracy drop value. Returns NaN if the level is not found.
    """
    try:
        idx = occlusion_levels.index(drop_level)
        accuracy_at_level = accuracy_points[idx]
        return initial_accuracy - accuracy_at_level
    except ValueError:
        return float('nan')  # Return Not-a-Number if the level isn't in our list
