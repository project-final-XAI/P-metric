import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel
except ImportError as exc:
    raise ImportError("Please install transformers: pip install transformers") from exc

import config
from attribution.base import AttributionMethod
from attribution.unet_based import U2NetSaliencyMethod


DEVICE = torch.device(
    config.DEVICE if hasattr(config, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu")
)

_MODEL_CACHE: dict[str, nn.Module] = {}
DINO_MODEL_NAME = "facebook/dinov2-with-registers-base"
_DINO_INPUT_SIZE = (224, 224)


def _ensure_dinov2():
    if "dinov2" not in _MODEL_CACHE:
        model = AutoModel.from_pretrained(DINO_MODEL_NAME)
        model.to(DEVICE).eval()
        _MODEL_CACHE["dinov2"] = model
    return _MODEL_CACHE["dinov2"]


class U2NetDinoFusionMethod(AttributionMethod):
    """Fuses U2Net saliency and DINO maps with fixed 50/50 weights."""

    def __init__(self, method_name: str = "u2net_dino_fusion") -> None:
        super().__init__(method_name)
        self._dinov2 = None
        self._u2net = U2NetSaliencyMethod()

    def _get_dinov2(self):
        if self._dinov2 is None:
            self._dinov2 = _ensure_dinov2()
        return self._dinov2

    def _detect_polarity_batch(self, batch_maps: torch.Tensor) -> torch.Tensor:
        """Flip maps where edge activation dominates center activation."""
        B, H, W = batch_maps.shape
        cy, cx = H // 2, W // 2
        center = batch_maps[:, cy - H // 8 : cy + H // 8, cx - W // 8 : cx + W // 8].mean(dim=(1, 2))
        edge = (
            batch_maps[:, : H // 10, :].mean(dim=(1, 2))
            + batch_maps[:, -H // 10 :, :].mean(dim=(1, 2))
        ) / 2
        flip_mask = (edge > center).float().view(B, 1, 1)
        return (1.0 - flip_mask) * batch_maps + flip_mask * (1.0 - batch_maps)

    def _compute_u2_map(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Use the existing U2Net attribution method implementation."""
        return self._u2net.compute(model, images, targets)

    def _compute_dino_map(self, images: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Compute DINO CLS-vs-patch cosine heatmap with polarity correction."""
        img_dino = F.interpolate(images, size=_DINO_INPUT_SIZE, mode="bilinear")
        dino = self._get_dinov2()
        with torch.no_grad():
            outputs = dino(img_dino)
            last_hidden = outputs.last_hidden_state
            num_regs = getattr(dino.config, "num_register_tokens", 0)
            cls_token = F.normalize(last_hidden[:, 0:1, :], dim=-1)
            patch_tokens = F.normalize(last_hidden[:, 1 + num_regs :, :], dim=-1)
            sims = (patch_tokens * cls_token).sum(dim=-1)
            side = int(np.sqrt(sims.shape[1]))
            dino_map = sims.reshape(images.shape[0], side, side)
            dino_map = F.interpolate(dino_map.unsqueeze(1), size=(height, width), mode="bilinear").squeeze(1)
            return self._detect_polarity_batch(dino_map)

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, _, height, width = images.shape
        u2_map = self._compute_u2_map(model, images, targets)
        dino_map = self._compute_dino_map(images, height, width)
        fused = 0.5 * u2_map + 0.5 * dino_map
        return self._normalize_attribution(fused)
