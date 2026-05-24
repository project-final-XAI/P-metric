from abc import ABC, abstractmethod
from typing import Tuple, Union
import torch
import torch.nn.functional as F


class AttributionMethod(ABC):
    """Base class for all attribution methods.

    Implements the Template Method pattern: ``compute()`` controls the
    full lifecycle (optional resize ➜ raw attribution ➜ normalization ➜
    upscale) while subclasses only implement ``_compute_raw``.
    """

    def __init__(self, name: str):
        self.name = name

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Template method — DO NOT OVERRIDE IN SUBCLASSES.

        1. Pre-process:  resize input if ``target_size`` is set.
        2. Strategy:     call ``_compute_raw`` (subclass hook).
        3. Normalize:    min-max per heatmap to [0, 1].
        4. Upscale:      restore original spatial dimensions if needed.
        """
        _, _, orig_h, orig_w = images.shape

        # 2. Raw attribution (subclass hook)
        raw_attribution = self._compute_raw(model, images, targets)

        # 3. Normalize
        normalized = self._normalize_attribution(raw_attribution)

        # 4. Upscale back to original resolution if needed
        if normalized.shape[-2:] != (orig_h, orig_w):
            normalized = normalized.unsqueeze(1)
            normalized = F.interpolate(normalized, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            normalized = normalized.squeeze(1)

        return normalized

    @abstractmethod
    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute raw, unnormalized attribution maps.

        Subclasses implement this. Input images are already resized to
        ``target_size`` if that was specified.
        """
        pass

    def _normalize_attribution(self, attribution: torch.Tensor) -> torch.Tensor:
        """Min-max normalize attribution batches to [0, 1]."""
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
    """Base for methods that don't need the classifier model (DINO, U2Net, etc.)."""

    def _compute_raw(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.compute_independent(images)

    @abstractmethod
    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        """Compute raw attribution using only the input images.

        Args:
            images: (B, C, H, W) — already resized to target_size if applicable.

        Returns:
            Raw attribution tensor before normalization.
        """
        pass