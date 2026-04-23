"""
Base classes for all attribution methods.

Defines the unified interface and category-specific subclasses:
  - AttributionMethod        — abstract root (model-dependent methods)
  - ModelIndependentMethod   — methods that ignore the classifier model
"""

from abc import ABC, abstractmethod
import torch


class AttributionMethod(ABC):
    """Base class for all attribution methods."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute attribution maps for a batch of images.

        Args:
            model: Neural network model
            images: Batch of images (B, C, H, W)
            targets: Target class indices (B,)

        Returns:
            Attribution heatmaps (B, H, W) normalized to [0, 1]
        """
        pass

    def _normalize_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Normalize attribution to [0, 1] range (same device as input)."""
        att_abs = torch.abs(attribution.detach())

        if att_abs.ndim == 4 and att_abs.shape[1] > 1:
            att_abs = torch.mean(att_abs, dim=1)
        elif att_abs.ndim == 4 and att_abs.shape[1] == 1:
            att_abs = att_abs.squeeze(1)

        normalized = []
        for heatmap in att_abs:
            min_val, max_val = heatmap.min(), heatmap.max()
            if max_val > min_val:
                normalized.append((heatmap - min_val) / (max_val - min_val))
            else:
                normalized.append(heatmap)

        return torch.stack(normalized)


class ModelIndependentMethod(AttributionMethod):
    """Base for methods that don't need the classifier model (DINO, U2Net, etc.).

    Subclasses implement ``compute_independent(images)`` instead of the full
    three-arg ``compute``.  The ``model`` and ``targets`` arguments are
    accepted for interface compatibility but never forwarded.
    """

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_independent(images)

    @abstractmethod
    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        """Compute attribution using only the input images.

        Args:
            images: Batch of images (B, C, H, W)

        Returns:
            Attribution heatmaps (B, H, W) normalized to [0, 1]
        """
        pass
