"""
Unified DINOv2 XAI Attribution Suite.

Provides three distinct attribution methods that share a single memory-efficient
HuggingFace backend:
1. Dinov2PcaGaussianMethod: Legacy PCA + Gaussian blob.
2. Dinov2AttentionMethod: Self-attention heatmap.
3. Dinov2UnifiedMethod: Advanced hybrid combining PCA and entropy-weighted attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel

import config
from attribution.base import AttributionMethod


# ---------------------------------------------------------------------------
# Constants & Setup
# ---------------------------------------------------------------------------

DEVICE = torch.device(
    getattr(config, "DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)

_PATCH_SIZE = 14
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared Utilities
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a numpy array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _soft_iou(fg: np.ndarray, attn: np.ndarray) -> float:
    """Soft IoU between a [0,1] foreground mask and a [0,1] attention map."""
    inter = (fg * attn).sum()
    union = (fg + attn - fg * attn).sum()
    return float(inter / (union + 1e-8))


def _attention_entropy(attn_head: np.ndarray) -> float:
    """Normalised Shannon entropy of an attention head's distribution [0, 1]."""
    p = attn_head.flatten()
    p = p / (p.sum() + 1e-8)
    h = -np.sum(p * np.log(p + 1e-12))
    h_max = np.log(len(p) + 1e-12)
    return float(h / (h_max + 1e-8))


def _create_soft_heatmap(binary_mask: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """Distance-transform + Gaussian blur → smooth XAI blob."""
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    heatmap = cv2.GaussianBlur(dist, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    mx = heatmap.max()
    if mx > 0:
        heatmap = (heatmap - heatmap.min()) / (mx - heatmap.min() + 1e-8)
    return heatmap


def _tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """Convert a normalized tensor batch back to PIL images."""
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    imgs = (images * std + mean).clamp(0, 1)
    imgs = (imgs * 255).byte().cpu()
    return [
        Image.fromarray(imgs[i].permute(1, 2, 0).numpy()) for i in range(imgs.shape[0])
    ]


def _border_mask(gh: int, gw: int, border: int) -> np.ndarray:
    """Boolean mask — True at border patches."""
    m = np.zeros((gh, gw), dtype=bool)
    m[:border, :] = m[-border:, :] = m[:, :border] = m[:, -border:] = True
    return m


def _center_mask(gh: int, gw: int, cf: float) -> np.ndarray:
    """Boolean mask — True for the central rectangle."""
    m = np.zeros((gh, gw), dtype=bool)
    h0, h1 = max(int(gh * (0.5 - cf / 2)), 0), min(int(gh * (0.5 + cf / 2)), gh)
    w0, w1 = max(int(gw * (0.5 - cf / 2)), 0), min(int(gw * (0.5 + cf / 2)), gw)
    m[h0:h1, w0:w1] = True
    return m


# ---------------------------------------------------------------------------
# Core Extractor Backend
# ---------------------------------------------------------------------------

@dataclass
class DinoContext:
    patch_features: np.ndarray   # (num_patches, hidden_dim)
    last_attention: np.ndarray   # (num_heads, seq_len, seq_len)
    grid_h: int
    grid_w: int
    orig_w: int
    orig_h: int
    num_skip: int                # Number of non-patch tokens (CLS + registers)


class Dinov2Extractor:
    """Singleton-style manager for the HuggingFace DINOv2 model and forward passes."""
    _CACHE: dict[str, tuple[AutoImageProcessor, AutoModel]] = {}

    @classmethod
    def get_model(cls, use_registers: bool) -> tuple[AutoImageProcessor, AutoModel]:
        key = "reg" if use_registers else "std"
        if key not in cls._CACHE:
            name = "facebook/dinov2-with-registers-small" if use_registers else "facebook/dinov2-base"
            print(f"[Dinov2Extractor] Loading {name} on {DEVICE} …")
            processor = AutoImageProcessor.from_pretrained(name)
            model = AutoModel.from_pretrained(name, output_attentions=True).to(DEVICE).eval()
            cls._CACHE[key] = (processor, model)
        return cls._CACHE[key]

    @classmethod
    def forward(cls, img_pil: Image.Image, use_registers: bool) -> DinoContext:
        processor, model = cls.get_model(use_registers)
        inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)

        h, w = inputs["pixel_values"].shape[2:]
        gh, gw = h // _PATCH_SIZE, w // _PATCH_SIZE
        num_patches = gh * gw

        with torch.no_grad():
            outputs = model(**inputs)

        seq_len = outputs.last_hidden_state.shape[1]
        num_skip = seq_len - num_patches

        # Extract patch features and attention
        feat_np = outputs.last_hidden_state[0, num_skip:, :].cpu().numpy()
        attn_np = outputs.attentions[-1][0].cpu().numpy()

        return DinoContext(
            patch_features=feat_np,
            last_attention=attn_np,
            grid_h=gh,
            grid_w=gw,
            orig_w=img_pil.width,
            orig_h=img_pil.height,
            num_skip=num_skip
        )


# ---------------------------------------------------------------------------
# Base Method Class
# ---------------------------------------------------------------------------

class Dinov2AttributionBase(AttributionMethod):
    """Base class handling batching and spatial resizing."""
    def __init__(self, name: str):
        super().__init__(name)
        self.use_registers = getattr(config, "DINO_USE_REGISTERS", True)

    def _heatmap_single(self, ctx: DinoContext) -> np.ndarray:
        raise NotImplementedError

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        pil_images = _tensor_batch_to_pil(images)

        heatmaps = []
        for pil_img in pil_images:
            ctx = Dinov2Extractor.forward(pil_img, self.use_registers)
            hmap = self._heatmap_single(ctx)
            # Ensure it is resized to the original PIL dimensions
            if hmap.shape != (ctx.orig_h, ctx.orig_w):
                hmap = cv2.resize(hmap, (ctx.orig_w, ctx.orig_h))
            heatmaps.append(hmap)

        heatmaps_np = np.stack(heatmaps, axis=0)

        # Resize to match the target tensor spatial dimensions if needed
        if heatmaps_np.shape[1] != H or heatmaps_np.shape[2] != W:
            ht = torch.from_numpy(heatmaps_np).unsqueeze(1).float()
            ht = F.interpolate(ht, size=(H, W), mode="bilinear", align_corners=False)
            heatmaps_np = ht.squeeze(1).numpy()

        return torch.from_numpy(heatmaps_np).float()


# ===========================================================================
# Method 1 – PCA + Gaussian
# ===========================================================================

class Dinov2PcaGaussianMethod(Dinov2AttributionBase):
    """Legacy method: PCA components + Center-vs-Border Heuristic."""
    def __init__(self):
        super().__init__("dinov2_pca_gaussian")
        self.n_components = getattr(config, "DINO_PCA_N_COMPONENTS", 5)
        self.threshold_q = getattr(config, "DINO_PCA_THRESHOLD_Q", 0.5)
        self.sigma = getattr(config, "DINO_PCA_SIGMA", 10.0)
        self.border = getattr(config, "DINO_PCA_BORDER", 1)
        self.center_frac = getattr(config, "DINO_PCA_CENTER_FRAC", 0.4)

    def _heatmap_single(self, ctx: DinoContext) -> np.ndarray:
        pca_feat = PCA(n_components=self.n_components).fit_transform(ctx.patch_features)

        bf = _border_mask(ctx.grid_h, ctx.grid_w, self.border).flatten()
        cf = _center_mask(ctx.grid_h, ctx.grid_w, self.center_frac).flatten()

        best_map, best_score = None, -np.inf
        n = min(self.n_components, pca_feat.shape[1])

        for i in range(n):
            for sign in (+1.0, -1.0):
                cand = _normalize(sign * pca_feat[:, i])
                score = cand[cf].mean() - cand[bf].mean()
                if score > best_score:
                    best_score = score
                    best_map = cand.reshape(ctx.grid_h, ctx.grid_w)

        binary_mask = (best_map > np.quantile(best_map, self.threshold_q)).astype(float)
        binary_full = cv2.resize(binary_mask, (ctx.orig_w, ctx.orig_h))
        return _create_soft_heatmap(binary_full, sigma=self.sigma)


# ===========================================================================
# Method 2 – CLS Self-Attention
# ===========================================================================

class Dinov2AttentionMethod(Dinov2AttributionBase):
    """Legacy method: Averaged CLS-to-patch self-attention."""
    def __init__(self):
        super().__init__("dinov2_attention")
        self.smooth_sigma = getattr(config, "DINO_ATTENTION_SMOOTH_SIGMA", 0.0)

    def _heatmap_single(self, ctx: DinoContext) -> np.ndarray:
        # ctx.last_attention shape: (num_heads, seq_len, seq_len)
        # Average heads, look at CLS token (index 0), ignore prefix tokens
        patch_attn = ctx.last_attention[:, 0, ctx.num_skip:].mean(axis=0)
        attn_map = _normalize(patch_attn.reshape(ctx.grid_h, ctx.grid_w))

        if self.smooth_sigma > 0:
            attn_map = _normalize(gaussian_filter(attn_map, sigma=self.smooth_sigma))

        return cv2.resize(attn_map, (ctx.orig_w, ctx.orig_h))


# ===========================================================================
# Method 3 – Unified Hybrid
# ===========================================================================

class Dinov2UnifiedMethod(Dinov2AttributionBase):
    """Advanced Hybrid: Attention-adjudicated PCA selection + Hybrid merging."""
    def __init__(self):
        super().__init__("dinov2_unified")
        self.mode = getattr(config, "DINO_METHOD_MODE", "hybrid")
        self.n_components = getattr(config, "DINO_PCA_N_COMPONENTS", 5)
        self.threshold_q = getattr(config, "DINO_PCA_THRESHOLD_Q", 0.45)
        self.sigma = getattr(config, "DINO_PCA_SIGMA", 10.0)
        self.border = getattr(config, "DINO_PCA_BORDER", 1)
        self.center_frac = getattr(config, "DINO_PCA_CENTER_FRAC", 0.4)
        self.attn_min_peak = getattr(config, "DINO_ATTN_MIN_PEAK", 0.25)
        self.gamma = getattr(config, "DINO_HYBRID_GAMMA", 0.7)

    def _extract_attention(self, ctx: DinoContext) -> np.ndarray:
        attn_heads = ctx.last_attention[:, 0, ctx.num_skip:]
        weights = np.array([1.0 - _attention_entropy(h) for h in attn_heads])

        w_sum = weights.sum()
        if w_sum < 1e-6: # Fallback to uniform if all heads are diffuse
            weights, w_sum = np.ones(len(weights)), float(len(weights))

        weighted_attn = (weights[:, None] * attn_heads).sum(0) / w_sum
        return _normalize(weighted_attn.reshape(ctx.grid_h, ctx.grid_w))

    def _heatmap_single(self, ctx: DinoContext) -> np.ndarray:
        attn_map = self._extract_attention(ctx)
        attn_full = cv2.resize(attn_map, (ctx.orig_w, ctx.orig_h))

        if self.mode == "attention":
            return _normalize(attn_full) ** self.gamma if self.gamma != 1.0 else _normalize(attn_full)

        pca_feat = PCA(n_components=self.n_components).fit_transform(ctx.patch_features)
        use_attn = attn_map.max() >= self.attn_min_peak

        best_map, best_q, best_score = None, self.threshold_q, -np.inf
        n = min(self.n_components, pca_feat.shape[1])

        bf = _border_mask(ctx.grid_h, ctx.grid_w, self.border).flatten()
        cf = _center_mask(ctx.grid_h, ctx.grid_w, self.center_frac).flatten()

        for i in range(n):
            for sign in (+1.0, -1.0):
                cand = _normalize(sign * pca_feat[:, i]).reshape(ctx.grid_h, ctx.grid_w)

                # Multi-scale check
                for q in (self.threshold_q, max(self.threshold_q - 0.15, 0.1)):
                    if use_attn:
                        fg = (cand > np.quantile(cand, q)).astype(float)
                        score = _soft_iou(fg, attn_map)
                    else:
                        flat = cand.flatten()
                        score = flat[cf].mean() - flat[bf].mean()

                    if score > best_score:
                        best_score, best_map, best_q = score, cand, q

        pca_binary = (best_map > np.quantile(best_map, best_q)).astype(np.float32)
        pca_binary_full = cv2.resize(pca_binary, (ctx.orig_w, ctx.orig_h))
        pca_soft = _create_soft_heatmap(pca_binary_full, sigma=self.sigma)

        if self.mode == "pca":
            return pca_soft ** self.gamma if self.gamma != 1.0 else pca_soft

        # Hybrid Mode
        fused = _normalize(pca_soft * attn_full)
        return fused ** self.gamma if self.gamma != 1.0 else fused