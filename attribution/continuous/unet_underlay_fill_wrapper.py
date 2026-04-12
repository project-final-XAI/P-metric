"""
U2Net-underlay wrapper for model-dependent XAI methods.

Creates a continuous fusion heatmap by using U2Net saliency as a base layer
and compositing the chosen XAI method as a fill layer.
"""

import torch

from attribution.base import AttributionMethod
from attribution.model_independent.unet_based import U2NetSaliencyMethod


class U2NetUnderlayFillWrapper(AttributionMethod):
    """Blend U2Net underlay with any model-dependent XAI fill map.

    Fusion rule:
        fused = underlay + (1 - underlay) * fill

    This keeps U2Net structure as the base while allowing the XAI map to fill
    in stronger evidence regions.
    """

    def __init__(
        self,
        base_method: AttributionMethod,
        name_suffix: str = "u2net_fill",
    ) -> None:
        super().__init__(f"{base_method.name}_{name_suffix}")
        self._base = base_method
        self._u2net = U2NetSaliencyMethod()

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        underlay = self._u2net.compute(model, images, targets)
        fill = self._base.compute(model, images, targets)

        underlay = self._normalize_attribution(underlay.unsqueeze(1)).squeeze(1)
        fill = self._normalize_attribution(fill.unsqueeze(1)).squeeze(1)

        fused = underlay + (1.0 - underlay) * fill
        return self._normalize_attribution(fused.unsqueeze(1)).squeeze(1)
