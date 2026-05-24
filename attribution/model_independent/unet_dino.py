"""
Fusion method: U2Net saliency + DINOv2 CLS-patch cosine with 50/50 blend.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter

try:
    from transformers import AutoModel, AutoImageProcessor
except ImportError as exc:
    raise ImportError("Please install transformers: pip install transformers") from exc

import config
from attribution.base import ModelIndependentMethod
from attribution._shared import DEVICE, get_cached_model
from attribution.model_independent.unet_based import U2NetSaliencyMethod
from attribution.model_independent.dinov2_methods import Dinov2TriSignalGuidedMethod

DINO_MODEL_NAME = getattr(config, "DINO_MODEL_NAME", "facebook/dinov2-with-registers-base")
PRODUCT_SMOOTHING = float(getattr(config, "DINO_PRODUCT_SMOOTHING", 0.05))

def normalize_map(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        return (arr - a_min) / (a_max - a_min)
    return np.zeros_like(arr, dtype=np.float64)

def detect_polarity(score_map: np.ndarray) -> np.ndarray:
    h, w = score_map.shape

    cy, cx = h // 2, w // 2
    radius = min(h, w) * 0.25
    ys, xs = np.ogrid[:h, :w]
    center_mask = ((ys - cy) ** 2 + (xs - cx) ** 2) <= radius ** 2

    margin = max(h, w) // 10
    edge_mask = np.zeros((h, w), dtype=bool)
    edge_mask[:margin, :] = True
    edge_mask[-margin:, :] = True
    edge_mask[:, :margin] = True
    edge_mask[:, -margin:] = True

    center_mean = score_map[center_mask].mean()
    edge_mean = score_map[edge_mask].mean()
    center_edge_vote = 1 if center_mean >= edge_mean else -1

    high_fraction = (score_map > 0.5).mean()
    compactness_vote = 1 if high_fraction <= 0.55 else -1

    if (center_edge_vote + compactness_vote) < 0:
        return 1.0 - score_map
    return score_map

def _ensure_dinov2_processor_and_model():
    def _load():
        attn_impl = getattr(config, "DINO_ATTN_IMPLEMENTATION", "eager")
        processor = AutoImageProcessor.from_pretrained(DINO_MODEL_NAME)
        model = AutoModel.from_pretrained(DINO_MODEL_NAME, attn_implementation=attn_impl)
        
        expected_regs = getattr(config, "DINO_NUM_REGISTERS", 4)
        actual_regs = getattr(model.config, "num_register_tokens", expected_regs)
        if actual_regs != expected_regs:
            raise ValueError(
                f"DINO register count mismatch: config={expected_regs}, model={actual_regs}. "
                "Update DINO_NUM_REGISTERS or DINO_MODEL_NAME in config.py."
            )
        model.to(DEVICE).eval()
        return processor, model

    return get_cached_model("dinov2_hf_with_processor", _load)


class U2NetDinoFusionMethod(ModelIndependentMethod):
    """Fuses U2Net saliency and DINO maps with fixed 50/50 weights."""

    def __init__(self, method_name: str = "u2net_dino_fusion") -> None:
        super().__init__(method_name)
        self._dinov2_processor = None
        self._dinov2_model = None
        self._u2net = U2NetSaliencyMethod()

    def _get_dinov2_processor_and_model(self):
        if self._dinov2_processor is None or self._dinov2_model is None:
            self._dinov2_processor, self._dinov2_model = _ensure_dinov2_processor_and_model()
        return self._dinov2_processor, self._dinov2_model

    def _compute_dino_map(self, img_np: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        processor, model = self._get_dinov2_processor_and_model()

        image = Image.fromarray(img_np).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state

        patch_size = model.config.patch_size
        num_register_tokens = getattr(model.config, "num_register_tokens", 0)

        pixel_values = inputs["pixel_values"]
        _, _, proc_h, proc_w = pixel_values.shape
        num_patches_h = proc_h // patch_size
        num_patches_w = proc_w // patch_size
        num_patches = num_patches_h * num_patches_w

        cls_token = last_hidden[:, 0:1, :]
        patch_tokens = last_hidden[:, 1 + num_register_tokens:, :]

        if patch_tokens.shape[1] != num_patches:
            raise ValueError(f"Patch-token count mismatch: got {patch_tokens.shape[1]}, expected {num_patches}")

        cls_token = F.normalize(cls_token, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        sims = (patch_tokens * cls_token).sum(dim=-1)

        patch_map = sims.reshape(num_patches_h, num_patches_w).detach().cpu().numpy()
        patch_map = normalize_map(patch_map)
        patch_map = detect_polarity(patch_map)

        patch_img = Image.fromarray((patch_map * 255).astype(np.uint8))
        heatmap = np.array(patch_img.resize((orig_w, orig_h), Image.BILINEAR), dtype=np.float64) / 255.0

        heatmap = gaussian_filter(heatmap, sigma=2.0)
        heatmap = normalize_map(heatmap)
        return heatmap

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, _, orig_h, orig_w = images.shape
        
        # `images` is ImageNet normalized. Un-normalize to [0,1]
        _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        rgb01 = (images * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)
        
        # Compute U2Net using the updated U2NetSaliencyMethod
        u2_maps = self._u2net.compute_independent(images) # Shape: (B, 1, H, W)
        
        fused_maps = []
        for i in range(B):
            img_np = (rgb01[i].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            
            dino_np = self._compute_dino_map(img_np, orig_h, orig_w)
            u2_np = u2_maps[i, 0].cpu().numpy()
            
            fused = (0.5 * u2_np) + (0.5 * dino_np)
            fused = gaussian_filter(fused, sigma=1.0)
            fused = normalize_map(fused)
            
            fused_maps.append(torch.from_numpy(fused).to(device=images.device, dtype=images.dtype).unsqueeze(0))
            
        return torch.stack(fused_maps, dim=0)


class U2NetDinoProductMethod(ModelIndependentMethod):
    """Fuses U2Net saliency and DINO maps via elementwise product."""

    def __init__(self, method_name: str = "u2net_dino_product") -> None:
        super().__init__(method_name)
        self._dinov2_processor = None
        self._dinov2_model = None
        self._u2net = U2NetSaliencyMethod()

    def _get_dinov2_processor_and_model(self):
        if self._dinov2_processor is None or self._dinov2_model is None:
            self._dinov2_processor, self._dinov2_model = _ensure_dinov2_processor_and_model()
        return self._dinov2_processor, self._dinov2_model

    def _compute_dino_map(self, img_np: np.ndarray, orig_h: int, orig_w: int) -> np.ndarray:
        processor, model = self._get_dinov2_processor_and_model()

        image = Image.fromarray(img_np).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden = outputs.last_hidden_state

        patch_size = model.config.patch_size
        num_register_tokens = getattr(model.config, "num_register_tokens", 0)

        pixel_values = inputs["pixel_values"]
        _, _, proc_h, proc_w = pixel_values.shape
        num_patches_h = proc_h // patch_size
        num_patches_w = proc_w // patch_size
        num_patches = num_patches_h * num_patches_w

        cls_token = last_hidden[:, 0:1, :]
        patch_tokens = last_hidden[:, 1 + num_register_tokens:, :]

        if patch_tokens.shape[1] != num_patches:
            raise ValueError(f"Patch-token count mismatch: got {patch_tokens.shape[1]}, expected {num_patches}")

        cls_token = F.normalize(cls_token, dim=-1)
        patch_tokens = F.normalize(patch_tokens, dim=-1)
        sims = (patch_tokens * cls_token).sum(dim=-1)

        patch_map = sims.reshape(num_patches_h, num_patches_w).detach().cpu().numpy()
        patch_map = normalize_map(patch_map)
        patch_map = detect_polarity(patch_map)

        patch_img = Image.fromarray((patch_map * 255).astype(np.uint8))
        heatmap = np.array(patch_img.resize((orig_w, orig_h), Image.BILINEAR), dtype=np.float64) / 255.0

        heatmap = gaussian_filter(heatmap, sigma=2.0)
        heatmap = normalize_map(heatmap)
        return heatmap

    def compute_independent(self, images: torch.Tensor) -> torch.Tensor:
        B, _, orig_h, orig_w = images.shape

        # `images` is ImageNet normalized. Un-normalize to [0,1]
        _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).to(device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        rgb01 = (images * _IMAGENET_STD + _IMAGENET_MEAN).clamp(0, 1)

        # Compute U2Net using the updated U2NetSaliencyMethod
        u2_maps = self._u2net.compute_independent(images)  # Shape: (B, 1, H, W)

        fused_maps = []
        for i in range(B):
            img_np = (rgb01[i].cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

            dino_np = self._compute_dino_map(img_np, orig_h, orig_w)
            u2_np = u2_maps[i, 0].cpu().numpy()

            smoothing = np.clip(PRODUCT_SMOOTHING, 0.0, 0.49)
            u2_smooth = (1.0 - smoothing) * u2_np + smoothing
            dino_smooth = (1.0 - smoothing) * dino_np + smoothing
            fused = u2_smooth * dino_smooth
            fused = gaussian_filter(fused, sigma=1.0)
            fused = normalize_map(fused)

            fused_maps.append(torch.from_numpy(fused).to(device=images.device, dtype=images.dtype).unsqueeze(0))

        return torch.stack(fused_maps, dim=0)


def _resize_np_float_map(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    img = Image.fromarray(arr.astype(np.float32), mode="F")
    return np.array(img.resize((out_w, out_h), Image.BILINEAR), dtype=np.float32)


def _norm01(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr, dtype=np.float32)


class Stage1StyleU2NetDinoHybridMethod(ModelIndependentMethod):
    """Stage1-style DINO+U2Net hybrid variants."""

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
        B, _, H, W = images.shape
        dino_maps = self._dino.compute_independent(images)   # (B, 1, H, W)
        u2_maps = self._u2net.compute_independent(images)    # (B, 1, H, W)

        out = []
        for i in range(B):
            dino_np = dino_maps[i, 0].detach().cpu().numpy().astype(np.float32)
            u2_np = u2_maps[i, 0].detach().cpu().numpy().astype(np.float32)

            dino_c = _resize_np_float_map(dino_np, self.combine_size, self.combine_size)
            u2_c = _resize_np_float_map(u2_np, self.combine_size, self.combine_size)

            if self.combine_mode == "clamped_sum":
                hm = np.clip(dino_c + u2_c, 0.0, 1.0).astype(np.float32)
            else:
                hm = _norm01(dino_c + u2_c)

            if hm.shape != (H, W):
                hm = _resize_np_float_map(hm, H, W)
                hm = _norm01(hm)

            out.append(torch.from_numpy(hm).to(device=images.device, dtype=images.dtype).unsqueeze(0))

        return torch.stack(out, dim=0)


class U2NetDinoAvg224Method(Stage1StyleU2NetDinoHybridMethod):
    """Stage1 `u2net+dino`: sum and normalize on 224 grid."""

    def __init__(self) -> None:
        super().__init__(
            method_name="u2net_dino_avg_224",
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
            method_name="dino_u2net_sum_224",
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
            method_name="u2net_dino_avg_320",
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
            method_name="dino448_u2net_avg_320",
            dino_input_size=448,
            dino_output_size=224,
            dino_hi_res_guided_filter=True,
            combine_size=320,
            combine_mode="norm_sum",
        )
