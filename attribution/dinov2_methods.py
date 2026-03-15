"""
DINOv2-based attribution methods.

Provides two attribution methods that integrate the DINOv2 heatmaps into the
standard Phase 1/2 pipeline:

- Dinov2PcaGaussianMethod: PCA + Gaussian soft heatmap (from method_pca.py)
- Dinov2AttentionMethod:   Self-attention heatmap  (from method_attention.py)

Both methods decide whether to use DINOv2 register models based on config flags:
- config.DINO_PCA_USE_REGISTERS       (bool, default True)
- config.DINO_ATTENTION_USE_REGISTERS (bool, default True)

Additional PCA tuning knobs (all optional in config):
- config.DINO_PCA_N_COMPONENTS  (int,   default 5)    – how many PCA components to evaluate
- config.DINO_PCA_THRESHOLD_Q   (float, default 0.5)  – quantile for binary foreground mask
- config.DINO_PCA_SIGMA         (float, default 10.0) – Gaussian blur strength
- config.DINO_PCA_BORDER        (int,   default 1)    – border width (patches) for scoring
- config.DINO_PCA_CENTER_FRAC   (float, default 0.4)  – center fraction for scoring

Additional Attention tuning knobs:
- config.DINO_ATTENTION_SMOOTH_SIGMA (float, default 0.0) – optional Gaussian smoothing
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model

import config
from attribution.base import AttributionMethod


# ---------------------------------------------------------------------------
# Device — pulled from config if available, otherwise auto-detect
# ---------------------------------------------------------------------------
DEVICE = torch.device(
    config.DEVICE
    if hasattr(config, "DEVICE")
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

# DINOv2 patch size is always 14 for all public checkpoints
_PATCH_SIZE = 14

# Fixed input size for the attention branch (must be divisible by 14)
_ATTN_IMAGE_SIZE = (518, 518)

# ImageNet normalization used by all DINOv2 variants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize a numpy array to [0, 1]."""
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _create_soft_heatmap(binary_mask: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Turn a binary mask into a smooth XAI-style heatmap.

    Pipeline:
      1. Distance Transform  → high values at object center, zero at background
      2. Gaussian blur       → smooth, natural-looking attribution blob
      3. Normalize to [0, 1]

    Using cv2.GaussianBlur (faster than scipy for 2-D arrays).
    ksize=(0,0) lets OpenCV auto-compute kernel size from sigma.
    """
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    dist = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    heatmap = cv2.GaussianBlur(dist, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    mx = heatmap.max()
    if mx > 0:
        heatmap = (heatmap - heatmap.min()) / (mx - heatmap.min() + 1e-8)
    return heatmap


def _tensor_batch_to_pil(images: torch.Tensor) -> list[Image.Image]:
    """
    Convert a (B, C, H, W) float tensor (ImageNet-normalized) back to PIL images.
    Used so the HuggingFace processor can re-encode them cleanly.
    """
    mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD,  device=images.device).view(1, 3, 1, 1)
    imgs_rgb = (images * std + mean).clamp(0, 1)          # un-normalize
    imgs_uint8 = (imgs_rgb * 255).byte().cpu()            # → uint8
    return [
        Image.fromarray(imgs_uint8[i].permute(1, 2, 0).numpy())
        for i in range(imgs_uint8.shape[0])
    ]


# ---------------------------------------------------------------------------
# Shared model caches  (one per method family to avoid cross-contamination)
# ---------------------------------------------------------------------------
_PCA_MODEL_CACHE:  dict[str, tuple] = {}   # key: "std" | "reg"  → (processor, model)
_ATTN_MODEL_CACHE: dict[str, object] = {}  # key: "std" | "reg"  → torch.hub model


# ===========================================================================
# Method 1 – PCA + Gaussian
# ===========================================================================

class Dinov2PcaGaussianMethod(AttributionMethod):
    """
    Attribution method using DINOv2 patch features + PCA + Gaussian soft heatmap.

    Pipeline per image
    ------------------
    1. Forward pass through DINOv2 (HuggingFace Transformers).
    2. Extract patch-token features from the last hidden state, dynamically
       skipping CLS + any register tokens.
    3. Run PCA over the first ``n_components`` components.
    4. Select the best (component, polarity) pair by "center-vs-border contrast":
       the component whose high values cluster in the image center (object) and
       low values live at the border (background) wins.
    5. Threshold at ``threshold_q`` quantile → binary foreground mask.
    6. Distance Transform + Gaussian blur → smooth soft heatmap.

    Configuration keys read from ``config``
    ----------------------------------------
    DINO_PCA_USE_REGISTERS  bool   True   – use facebook/dinov2-with-registers-small
    DINO_PCA_N_COMPONENTS   int    5      – PCA components to evaluate
    DINO_PCA_THRESHOLD_Q    float  0.5    – foreground quantile threshold
    DINO_PCA_SIGMA          float  10.0   – Gaussian blur strength
    DINO_PCA_BORDER         int    1      – border-patch width for scoring
    DINO_PCA_CENTER_FRAC    float  0.4    – center-region fraction for scoring
    """

    def __init__(self) -> None:
        super().__init__("dinov2_pca_gaussian")

        # Pull all knobs from config, with safe defaults
        self.use_registers:  bool  = getattr(config, "DINO_PCA_USE_REGISTERS", True)
        self.n_components:   int   = getattr(config, "DINO_PCA_N_COMPONENTS",  5)
        self.threshold_q:    float = getattr(config, "DINO_PCA_THRESHOLD_Q",   0.5)
        self.sigma:          float = getattr(config, "DINO_PCA_SIGMA",         10.0)
        self.border:         int   = getattr(config, "DINO_PCA_BORDER",        1)
        self.center_frac:    float = getattr(config, "DINO_PCA_CENTER_FRAC",   0.4)

        # Lazily loaded in _ensure_model()
        self._processor: Optional[AutoImageProcessor] = None
        self._dino_model: Optional[Dinov2Model] = None

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load and cache the DINOv2 HuggingFace model (called once on first use)."""
        cache_key = "reg" if self.use_registers else "std"
        if cache_key not in _PCA_MODEL_CACHE:
            model_name = (
                "facebook/dinov2-with-registers-small"
                if self.use_registers
                else "facebook/dinov2-base"
            )
            print(f"[Dinov2PcaGaussianMethod] Loading {model_name} on {DEVICE} …")
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = Dinov2Model.from_pretrained(model_name).to(DEVICE)
            model.eval()
            _PCA_MODEL_CACHE[cache_key] = (processor, model)

        self._processor, self._dino_model = _PCA_MODEL_CACHE[cache_key]

    # ------------------------------------------------------------------
    # Internal: component selection (the robust inversion logic)
    # ------------------------------------------------------------------

    @staticmethod
    def _border_mask(grid_h: int, grid_w: int, border: int) -> np.ndarray:
        """Boolean mask — True at border patches (likely background)."""
        m = np.zeros((grid_h, grid_w), dtype=bool)
        m[:border, :]  = True
        m[-border:, :] = True
        m[:, :border]  = True
        m[:, -border:] = True
        return m

    @staticmethod
    def _center_mask(grid_h: int, grid_w: int, center_frac: float) -> np.ndarray:
        """Boolean mask — True for the central rectangle (likely object)."""
        m  = np.zeros((grid_h, grid_w), dtype=bool)
        h0 = max(int(grid_h * (0.5 - center_frac / 2)), 0)
        h1 = min(int(grid_h * (0.5 + center_frac / 2)), grid_h)
        w0 = max(int(grid_w * (0.5 - center_frac / 2)), 0)
        w1 = min(int(grid_w * (0.5 + center_frac / 2)), grid_w)
        m[h0:h1, w0:w1] = True
        return m

    def _select_best_component(
        self,
        pca_features: np.ndarray,
        grid_h: int,
        grid_w: int,
    ) -> tuple[np.ndarray, int, float]:
        """
        Evaluate every component × polarity pair and return the one that best
        separates the image center from the image border.

        Returns
        -------
        best_map   : (grid_h, grid_w) normalized foreground map in [0, 1]
        best_comp  : 0-based index of the winning PCA component
        best_score : center_mean - border_mean score of the winner
        """
        border_flat = self._border_mask(grid_h, grid_w, self.border).flatten()
        center_flat = self._center_mask(grid_h, grid_w, self.center_frac).flatten()

        best_map:   Optional[np.ndarray] = None
        best_score: float = -np.inf
        best_comp:  int   = 0

        n = min(self.n_components, pca_features.shape[1])
        for i in range(n):
            for sign in (+1.0, -1.0):
                candidate = _normalize(sign * pca_features[:, i])
                score = candidate[center_flat].mean() - candidate[border_flat].mean()
                if score > best_score:
                    best_score = score
                    best_map   = candidate.reshape(grid_h, grid_w)
                    best_comp  = i

        return best_map, best_comp, best_score  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal: single-image heatmap
    # ------------------------------------------------------------------

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        """Return a (H, W) soft heatmap in [0,1] for one PIL image."""
        inputs = self._processor(images=img_pil, return_tensors="pt").to(DEVICE)

        h = inputs["pixel_values"].shape[2]
        w = inputs["pixel_values"].shape[3]
        grid_h = h // _PATCH_SIZE
        grid_w = w // _PATCH_SIZE
        num_patches = grid_h * grid_w

        with torch.no_grad():
            outputs = self._dino_model(**inputs)
            seq_len = outputs.last_hidden_state.shape[1]

            num_extra = seq_len - num_patches
            if num_extra < 0:
                raise RuntimeError(
                    f"[Dinov2PcaGaussianMethod] Token count mismatch: "
                    f"seq_len={seq_len}, expected_patches={num_patches}"
                )
            # Skip CLS + register tokens; keep only patch tokens
            features = outputs.last_hidden_state[:, num_extra:, :]

        features_np = features.squeeze(0).cpu().numpy()  # (num_patches, hidden_dim)

        pca = PCA(n_components=self.n_components)
        pca_features = pca.fit_transform(features_np)

        pca_map, best_comp, score = self._select_best_component(
            pca_features, grid_h, grid_w
        )
        print(
            f"[Dinov2PcaGaussianMethod] PC{best_comp + 1} selected  "
            f"(center-border score={score:.4f})"
        )

        # Binary foreground mask → resize to original image dimensions
        binary_mask = (pca_map > np.quantile(pca_map, self.threshold_q)).astype(float)
        W_orig, H_orig = img_pil.size                          # PIL: (W, H)
        binary_mask = cv2.resize(binary_mask, (W_orig, H_orig))

        return _create_soft_heatmap(binary_mask, sigma=self.sigma)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        model,                      # unused — DINOv2 is self-contained
        images: torch.Tensor,       # (B, C, H, W) ImageNet-normalized
        targets: torch.Tensor,      # (B,) — unused (unsupervised method)
    ) -> torch.Tensor:
        """
        Compute PCA-based Gaussian heatmaps for a batch of images.

        Args:
            model:   Unused (required by AttributionMethod interface).
            images:  Batch tensor (B, C, H, W), ImageNet-normalized float32.
            targets: Target class indices (B,) — not used by this method.

        Returns:
            Tensor of shape (B, H, W), dtype float32, values in [0, 1].
            The spatial dimensions match the input ``images``.
        """
        self._ensure_model()

        B, C, H, W = images.shape
        pil_images = _tensor_batch_to_pil(images)

        heatmaps = np.stack(
            [self._heatmap_single(pil_img) for pil_img in pil_images],
            axis=0,
        )  # (B, H_pil, W_pil)  — H_pil may differ from H if processor rescaled

        # Resize to match the original tensor spatial dimensions if needed
        if heatmaps.shape[1] != H or heatmaps.shape[2] != W:
            heatmaps_t = torch.from_numpy(heatmaps).unsqueeze(1).float()  # (B,1,h,w)
            heatmaps_t = F.interpolate(
                heatmaps_t, size=(H, W), mode="bilinear", align_corners=False
            )
            heatmaps = heatmaps_t.squeeze(1).numpy()                       # (B,H,W)

        return torch.from_numpy(heatmaps).float()


# ===========================================================================
# Method 2 – CLS Self-Attention
# ===========================================================================

class Dinov2AttentionMethod(AttributionMethod):
    """
    Attribution method using DINOv2 CLS-token self-attention maps.

    Pipeline per image
    ------------------
    1. Resize & normalize the image to 518×518 (optimal for DINOv2 patch grid).
    2. Forward pass through DINOv2 (torch.hub), intercepting the last block's
       self-attention matrix.
    3. Average across all heads → CLS-token attention vector over all tokens.
    4. Discard non-patch tokens (CLS + registers); take the last num_patches
       entries and reshape to (grid_h, grid_w).
    5. Optionally smooth with a Gaussian blur.
    6. Resize to original image dimensions.

    Configuration keys read from ``config``
    ----------------------------------------
    DINO_ATTENTION_USE_REGISTERS  bool   True  – use dinov2_vits14_reg
    DINO_ATTENTION_SMOOTH_SIGMA   float  0.0   – Gaussian blur on attention map
                                                  (0 = off; try 2–4 for softer look)
    """

    # ImageNet normalization transform for the attention branch
    _transform = transforms.Compose([
        transforms.Resize(_ATTN_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    def __init__(self) -> None:
        super().__init__("dinov2_attention")

        self.use_registers:  bool  = getattr(config, "DINO_ATTENTION_USE_REGISTERS", True)
        self.smooth_sigma:   float = getattr(config, "DINO_ATTENTION_SMOOTH_SIGMA",  0.0)

        self._dino_model = None  # lazily loaded

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load and cache the DINOv2 torch.hub model (called once on first use)."""
        cache_key = "reg" if self.use_registers else "std"
        if cache_key not in _ATTN_MODEL_CACHE:
            model_name = "dinov2_vits14_reg" if self.use_registers else "dinov2_vits14"
            print(f"[Dinov2AttentionMethod] Loading {model_name} on {DEVICE} …")
            m = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
            m.to(DEVICE)
            m.eval()
            _ATTN_MODEL_CACHE[cache_key] = m

        self._dino_model = _ATTN_MODEL_CACHE[cache_key]

    # ------------------------------------------------------------------
    # Internal: attention extraction
    # ------------------------------------------------------------------

    def _extract_attention(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract the last-block self-attention tensor for a pre-processed batch.

        Args:
            img_batch: (B, C, H, W) tensor already resized & normalized to
                       _ATTN_IMAGE_SIZE and on DEVICE.

        Returns:
            attn: (B, num_heads, N, N) attention tensor on DEVICE.
                  N = 1 + num_registers + num_patches  (all tokens).
        """
        model = self._dino_model

        with torch.no_grad():
            # Prefer the native helper when available (cleaner, model-version agnostic)
            if hasattr(model, "get_last_selfattention"):
                return model.get_last_selfattention(img_batch)

            # Manual fallback: run all blocks, intercept QK in the last one
            x = model.prepare_tokens_with_masks(img_batch)
            for i, blk in enumerate(model.blocks):
                if i < len(model.blocks) - 1:
                    x = blk(x)
                else:
                    # Last block: compute attention weights manually
                    x_norm = blk.norm1(x)
                    qkv    = blk.attn.qkv(x_norm)
                    B, N, C = x_norm.shape
                    nh  = blk.attn.num_heads
                    hd  = C // nh
                    qkv = qkv.reshape(B, N, 3, nh, hd).permute(2, 0, 3, 1, 4)
                    q, k = qkv[0], qkv[1]
                    attn = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
                    return attn.softmax(dim=-1)   # (B, nh, N, N)

        raise RuntimeError("[Dinov2AttentionMethod] Could not extract attention.")

    # ------------------------------------------------------------------
    # Internal: single-image heatmap
    # ------------------------------------------------------------------

    def _heatmap_single(self, img_pil: Image.Image) -> np.ndarray:
        """Return a (H_orig, W_orig) attention map in [0,1] for one PIL image."""
        W_orig, H_orig = img_pil.size

        img_tensor = self._transform(img_pil).unsqueeze(0).to(DEVICE)  # (1, C, 518, 518)

        attn = self._extract_attention(img_tensor)   # (1, nh, N, N)

        # Average over heads; CLS attends from position 0
        attn_cls = attn[0, :, 0, :].mean(dim=0)     # (N,)

        # Patch grid dimensions from the actual tensor size
        grid_h = img_tensor.shape[2] // _PATCH_SIZE
        grid_w = img_tensor.shape[3] // _PATCH_SIZE
        num_patches = grid_h * grid_w

        # Non-patch tokens (CLS + registers) appear at the front of the sequence.
        # Patch tokens are always the last num_patches entries.
        patch_attn = attn_cls[-num_patches:]                             # (num_patches,)
        attn_map   = patch_attn.reshape(grid_h, grid_w).cpu().numpy()
        attn_map   = _normalize(attn_map)

        # Optional Gaussian smoothing for a softer, XAI-like appearance
        if self.smooth_sigma > 0:
            attn_map = gaussian_filter(attn_map, sigma=self.smooth_sigma)
            attn_map = _normalize(attn_map)

        # Resize to original image spatial dimensions
        return cv2.resize(attn_map, (W_orig, H_orig))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def compute(
        self,
        model,                      # unused — DINOv2 is self-contained
        images: torch.Tensor,       # (B, C, H, W) ImageNet-normalized
        targets: torch.Tensor,      # (B,) — unused
    ) -> torch.Tensor:
        """
        Compute attention-based heatmaps for a batch of images.

        Args:
            model:   Unused (required by AttributionMethod interface).
            images:  Batch tensor (B, C, H, W), ImageNet-normalized float32.
            targets: Target class indices (B,) — not used by this method.

        Returns:
            Tensor of shape (B, H, W), dtype float32, values in [0, 1].
            The spatial dimensions match the input ``images``.
        """
        self._ensure_model()

        B, C, H, W = images.shape
        pil_images = _tensor_batch_to_pil(images)

        heatmaps = np.stack(
            [self._heatmap_single(pil_img) for pil_img in pil_images],
            axis=0,
        )  # (B, H_orig, W_orig)

        # Resize to match the original tensor spatial dimensions if needed
        if heatmaps.shape[1] != H or heatmaps.shape[2] != W:
            heatmaps_t = torch.from_numpy(heatmaps).unsqueeze(1).float()  # (B,1,h,w)
            heatmaps_t = F.interpolate(
                heatmaps_t, size=(H, W), mode="bilinear", align_corners=False
            )
            heatmaps = heatmaps_t.squeeze(1).numpy()                       # (B,H,W)

        return torch.from_numpy(heatmaps).float()