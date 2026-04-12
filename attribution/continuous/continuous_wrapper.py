"""
Generic wrapper that applies spatial smoothing to any attribution method.

Usage:
    base = IntegratedGradientsMethod()
    smooth = ContinuousWrapper(base, sigma=2.0)
    heatmap = smooth.compute(model, images, targets)
"""

import math
import torch
import torch.nn.functional as F

from attribution.base import AttributionMethod


class ContinuousWrapper(AttributionMethod):
    """Wraps any attribution method and applies Gaussian smoothing.

    Args:
        base_method: The attribution method to wrap.
        sigma:       Gaussian std-dev in pixels (default 2.0).
        name_suffix: Appended to the base method name for registry keys.
    """

    def __init__(
        self,
        base_method: AttributionMethod,
        sigma: float = 2.0,
        name_suffix: str = "continuous",
    ) -> None:
        super().__init__(f"{base_method.name}_{name_suffix}")
        self._base = base_method
        self.sigma = sigma

    def _gaussian_smooth(self, heatmaps: torch.Tensor) -> torch.Tensor:
        """Apply 2D Gaussian blur to (B, H, W) heatmaps."""
        if self.sigma <= 0:
            return heatmaps

        half = math.ceil(3.0 * self.sigma)
        ksize = 2 * half + 1
        coords = torch.arange(ksize, dtype=torch.float32) - half
        g = torch.exp(-(coords ** 2) / (2.0 * self.sigma ** 2))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        kernel = (kernel / kernel.sum()).view(1, 1, ksize, ksize)

        x = heatmaps.unsqueeze(1).float()  # (B, 1, H, W)
        x = F.pad(x, (half, half, half, half), mode="reflect")
        x = F.conv2d(x, kernel)
        return x.squeeze(1)

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        raw = self._base.compute(model, images, targets)
        smoothed = self._gaussian_smooth(raw)
        return self._normalize_attribution(smoothed.unsqueeze(1)).squeeze(1)
