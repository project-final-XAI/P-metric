"""
Fusion method: U2Net saliency + DINOv2 hybrid combinations.

Strictly implements the 4 Stage-1 fusion strategies:
- u2net+dino (avg 224)
- dino+u2net_sum (clamped sum 224)
- u2net+dino_320 (avg 320)
- dino448+u2net (hi-res guided avg 320)

Fully vectorized for batched GPU acceleration (zero NumPy/SciPy/PIL).
"""

import torch
import torch.nn.functional as F

from attribution.base import ModelIndependentMethod
from attribution.model_independent.unet_based import U2NetSaliencyMethod
from attribution.model_independent.dinov2_methods import Dinov2TriSignalGuidedMethod


def _norm01_tensor(t: torch.Tensor) -> torch.Tensor:
    """Batch-safe Min-Max Normalization across spatial dimensions."""
    b_min = t.amin(dim=(2, 3), keepdim=True)
    b_max = t.amax(dim=(2, 3), keepdim=True)
    diff = b_max - b_min
    diff = torch.where(diff > 0, diff, torch.ones_like(diff) * 1e-8)
    return (t - b_min) / diff


class Stage1StyleU2NetDinoHybridMethod(ModelIndependentMethod):
    """
    Core engine handling DINO+U2Net combinations safely within batches.
    Provides precise control over at which resolution the two tensors are merged.
    """
    def __init__(
        self,
        method_name: str,
        dino_input_size: int = 448,
        dino_output_size: int = 224,
        dino_hi_res_guided_filter: bool = False,
        combine_size: int = 224,
        combine_mode: str = "norm_sum",
    ) -> None:
        super().__init__(method_name)

        # Initialize the underlying independent methods
        self._u2net = U2NetSaliencyMethod()
        self._dino = Dinov2TriSignalGuidedMethod(
            dino_input_size=dino_input_size,
            output_size=dino_output_size,
            hi_res_guided_filter=dino_hi_res_guided_filter,
            guided_filter_radius=8,
            guided_filter_eps=1e-2,
            use_flip_tta=True,
        )
        self.combine_size = int(combine_size)
        self.combine_mode = combine_mode

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape

        # 1. Fetch batched heatmaps from isolated methods
        dino_maps = self._dino.compute_independent(images)
        u2_maps = self._u2net.compute_independent(images)

        # 2. Re-grid both to the requested combination spatial resolution
        if dino_maps.shape[-1] != self.combine_size:
            dino_c = F.interpolate(dino_maps, size=(self.combine_size, self.combine_size), mode="bilinear", align_corners=False)
        else:
            dino_c = dino_maps

        if u2_maps.shape[-1] != self.combine_size:
            u2_c = F.interpolate(u2_maps, size=(self.combine_size, self.combine_size), mode="bilinear", align_corners=False)
        else:
            u2_c = u2_maps

        # 3. Combine mathematically
        if self.combine_mode == "clamped_sum":
            hm = torch.clamp(dino_c + u2_c, 0.0, 1.0)
        else:  # norm_sum
            hm = _norm01_tensor(dino_c + u2_c)

        # 4. Upscale/Downscale back to framework target
        if hm.shape[-2:] != (H, W):
            hm = F.interpolate(hm, size=(H, W), mode="bilinear", align_corners=False)
            hm = _norm01_tensor(hm)

        return hm


# ===========================================================================
# The 4 Stage-1 Configurations
# ===========================================================================

class U2NetDinoAvg224Method(Stage1StyleU2NetDinoHybridMethod):
    """Stage1 `u2net+dino`: sum and normalize on 224 grid."""
    def __init__(self) -> None:
        super().__init__(
            method_name="u2net+dino",
            dino_input_size=448,
            dino_output_size=224,
            dino_hi_res_guided_filter=False,
            combine_size=224,
            combine_mode="norm_sum",
        )


class DINOU2NetClampedSum224Method(Stage1StyleU2NetDinoHybridMethod):
    """Stage1 `dino+u2net_sum`: clipped sum on 224 grid."""
    def __init__(self) -> None:
        super().__init__(
            method_name="dino+u2net_sum",
            dino_input_size=448,
            dino_output_size=224,
            dino_hi_res_guided_filter=False,
            combine_size=224,
            combine_mode="clamped_sum",
        )


class U2NetDinoAvg320Method(Stage1StyleU2NetDinoHybridMethod):
    """Stage1 `u2net+dino_320`: sum and normalize on 320 grid."""
    def __init__(self) -> None:
        super().__init__(
            method_name="u2net+dino_320",
            dino_input_size=448,
            dino_output_size=224,
            dino_hi_res_guided_filter=False,
            combine_size=320,
            combine_mode="norm_sum",
        )


class DINO448U2NetAvg320Method(Stage1StyleU2NetDinoHybridMethod):
    """Stage1 `dino448+u2net`: hi-res guided DINO + U2Net on 320 grid."""
    def __init__(self) -> None:
        super().__init__(
            method_name="dino448+u2net",
            dino_input_size=448,
            dino_output_size=224,
            dino_hi_res_guided_filter=True,
            combine_size=320,
            combine_mode="norm_sum",
        )