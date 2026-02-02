"""
DINOv2-based attribution methods.

Provides two attribution methods that integrate the DINOv2 heatmaps into the
standard Phase 1/2 pipeline:

- Dinov2PcaGaussianMethod: PCA + Gaussian soft heatmap (from dinov2_pca_gaussian_heatmap.py)
- Dinov2AttentionMethod:   Self-attention heatmap (from dinov2_attention_registers_heatmap.py)

Both methods decide whether to use DINOv2 register models based on config flags:
- config.DINO_PCA_USE_REGISTERS
- config.DINO_ATTENTION_USE_REGISTERS
"""

from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoImageProcessor, Dinov2Model

import config
from attribution.base import AttributionMethod


DEVICE = torch.device(config.DEVICE if hasattr(config, "DEVICE") else ("cuda" if torch.cuda.is_available() else "cpu"))


def _create_soft_heatmap(binary_mask: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """
    Turn a binary mask into a soft heatmap using distance transform + Gaussian blur.
    """
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    # Distance from background → high in object center
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    # Smooth to look like a soft attribution map
    heatmap = cv2.GaussianBlur(dist_transform, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    # Normalize to [0, 1]
    max_val = heatmap.max()
    if max_val > 0:
        heatmap = (heatmap - heatmap.min()) / (max_val - heatmap.min() + 1e-8)
    return heatmap


class Dinov2PcaGaussianMethod(AttributionMethod):
    """
    Attribution method using DINOv2 patch features + PCA + Gaussian soft heatmap.

    Configuration:
        config.DINO_PCA_USE_REGISTERS (bool) – if True, uses a DINOv2 registers model.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_pca_gaussian")
        self._processor: Optional[AutoImageProcessor] = None
        self._model: Optional[Dinov2Model] = None
        # Use a simple ToPIL transform to convert dataset tensors to images for the processor
        self._to_pil = transforms.ToPILImage()
        # Target heatmap size must match occlusion image shape (224x224)
        self._heatmap_size = (224, 224)

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        use_registers = getattr(config, "DINO_PCA_USE_REGISTERS", False)
        # Match the comparison script behavior:
        # - Without registers: standard DINOv2 base.
        # - With registers:   DINOv2-with-registers small checkpoint.
        if use_registers:
            model_name = "facebook/dinov2-with-registers-small"
        else:
            model_name = "facebook/dinov2-base"

        print(f"[Dinov2PcaGaussianMethod] Loading model: {model_name} (registers={use_registers}) on {DEVICE}")
        self._processor = AutoImageProcessor.from_pretrained(model_name)
        self._model = Dinov2Model.from_pretrained(model_name).to(DEVICE)
        self._model.eval()

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute PCA-based Gaussian heatmaps for a batch of images.

        Args:
            model: Unused (required by interface; DINOv2 is internal here)
            images: Batch of images (B, C, H, W)
            targets: Target class indices (B,) – unused for this unsupervised-style method

        Returns:
            Heatmaps tensor of shape (B, H, W) normalized to [0, 1]
        """
        del model, targets  # Not used – attribution is unsupervised relative to class
        self._ensure_model()

        assert self._processor is not None and self._model is not None

        images_cpu = images.detach().cpu()
        batch_size = images_cpu.shape[0]

        # Convert to list of PIL Images for the HuggingFace processor
        pil_images = [self._to_pil(img) for img in images_cpu]

        # DINOv2 processor handles resizing/normalization
        inputs = self._processor(images=pil_images, return_tensors="pt").to(DEVICE)

        # Infer expected patch grid from actual processed resolution
        # inputs["pixel_values"]: (B, C, H, W)
        h = inputs["pixel_values"].shape[2]
        w = inputs["pixel_values"].shape[3]
        patch_size = getattr(self._model.config, "patch_size", 14)
        grid_h = h // patch_size
        grid_w = w // patch_size
        num_patches_expected = grid_h * grid_w

        with torch.no_grad():
            outputs = self._model(**inputs)
            # last_hidden_state: (B, seq_len, D) where seq_len may include
            # CLS, optional register tokens, and patch tokens.
            seq_len = outputs.last_hidden_state.shape[1]
            num_extra_tokens = seq_len - num_patches_expected
            if num_extra_tokens < 0:
                raise RuntimeError(
                    f"[Dinov2PcaGaussianMethod] Unexpected token layout: "
                    f"seq_len={seq_len}, expected_patches={num_patches_expected}"
                )
            # Slice off CLS + any register tokens; keep only spatial patches
            features = outputs.last_hidden_state[:, num_extra_tokens:, :]

        features_np = features.cpu().numpy()  # (B, num_patches, D)

        heatmaps: list[np.ndarray] = []

        for b in range(batch_size):
            feats = features_np[b]  # (num_patches, D)

            # PCA to isolate object vs background along first component
            pca = PCA(n_components=3)
            pca_features = pca.fit_transform(feats)
            # Reshape using the calculated grid dimensions
            obj_mask = pca_features[:, 0].reshape(grid_h, grid_w)

            # Normalize mask
            obj_mask = (obj_mask - obj_mask.min()) / (obj_mask.max() - obj_mask.min() + 1e-8)

            # Ensure object is bright and background dark
            if obj_mask[0, 0] > 0.5:
                obj_mask = 1.0 - obj_mask

            # Binary mask via high-percentile threshold
            binary_mask = (obj_mask > np.quantile(obj_mask, 0.85)).astype(float)

            # Resize to target heatmap size (must align with occlusion resolution)
            binary_mask = cv2.resize(binary_mask, self._heatmap_size[::-1])

            # Final soft heatmap
            final_heatmap = _create_soft_heatmap(binary_mask)
            heatmaps.append(final_heatmap.astype(np.float32))

        heatmaps_np = np.stack(heatmaps, axis=0)  # (B, H, W)
        heatmaps_tensor = torch.from_numpy(heatmaps_np)

        # Final normalization to [0, 1] per heatmap
        return self._normalize_attribution(heatmaps_tensor)


class Dinov2AttentionMethod(AttributionMethod):
    """
    Attribution method using DINOv2 self-attention maps.

    Configuration:
        config.DINO_ATTENTION_USE_REGISTERS (bool) – if True, uses the DINOv2 register model.
    """

    def __init__(self) -> None:
        super().__init__("dinov2_attention")
        self._model: Optional[torch.nn.Module] = None
        self._model_name: Optional[str] = None
        self._patch_count: Optional[int] = None
        self._transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self._to_pil = transforms.ToPILImage()
        self._heatmap_size = (224, 224)

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        use_registers = getattr(config, "DINO_ATTENTION_USE_REGISTERS", False)
        self._model_name = "dinov2_vits14_reg" if use_registers else "dinov2_vits14"

        print(f"[Dinov2AttentionMethod] Loading model: {self._model_name} (registers={use_registers}) on {DEVICE}")
        # Load from torch.hub (Meta official DINOv2 repo)
        self._model = torch.hub.load("facebookresearch/dinov2", self._model_name, verbose=False)
        self._model.to(DEVICE)
        self._model.eval()

        # Try to read the number of spatial patch tokens from the model
        patch_count = None
        try:
            if hasattr(self._model, "patch_embed") and hasattr(self._model.patch_embed, "num_patches"):
                patch_count = int(self._model.patch_embed.num_patches)
        except Exception:
            patch_count = None

        # Fallback to 16x16 (ViT-S/14 @ 224x224) if not available
        if patch_count is None:
            patch_count = 16 * 16
        self._patch_count = patch_count

    def _extract_attention(self, img_batch: torch.Tensor) -> torch.Tensor:
        """
        Extract last-layer self-attention for a batch of images.

        Returns:
            attn: Tensor of shape (B, num_heads, num_tokens, num_tokens)
        """
        assert self._model is not None

        with torch.no_grad():
            if hasattr(self._model, "get_last_selfattention"):
                attn = self._model.get_last_selfattention(img_batch)
            else:
                # Manual fallback based on DINOv2 architecture
                x = self._model.prepare_tokens_with_masks(img_batch)
                for i, blk in enumerate(self._model.blocks):
                    if i < len(self._model.blocks) - 1:
                        x = blk(x)
                    else:
                        x_norm = blk.norm1(x)
                        qkv = blk.attn.qkv(x_norm)
                        B, N, C = x_norm.shape
                        num_heads = blk.attn.num_heads
                        head_dim = C // num_heads
                        qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)
                        q, k = qkv[0], qkv[1]
                        attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
                        attn = attn.softmax(dim=-1)

        return attn

    def compute(self, model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Compute attention-based heatmaps for a batch of images.

        Args:
            model: Unused (required by interface; DINOv2 is internal here)
            images: Batch of images (B, C, H, W)
            targets: Target class indices (B,) – unused

        Returns:
            Heatmaps tensor of shape (B, H, W) normalized to [0, 1]
        """
        del model, targets  # Not used – attention is unsupervised wrt class
        self._ensure_model()
        assert self._model is not None

        images_cpu = images.detach().cpu()
        batch_size = images_cpu.shape[0]

        # Prepare batch for DINOv2: resize + normalize
        pil_images = [self._to_pil(img) for img in images_cpu]
        dino_tensors = [self._transform(img) for img in pil_images]
        img_batch = torch.stack(dino_tensors, dim=0).to(DEVICE)

        attn = self._extract_attention(img_batch)
        # attn: (B, num_heads, num_tokens, num_tokens)

        heatmaps: list[np.ndarray] = []
        _, _, num_tokens, _ = attn.shape

        # Use known patch count (excludes CLS + possible register tokens)
        patch_count = self._patch_count or (num_tokens - 1)
        if patch_count > num_tokens - 1:
            patch_count = num_tokens - 1

        for b in range(batch_size):
            attn_b = attn[b]  # (num_heads, num_tokens, num_tokens)

            # Global attention from CLS token to all tokens, averaged over heads
            attn_cls = attn_b[:, 0, :].mean(dim=0)  # (num_tokens,)

            # Take only the last `patch_count` entries, which correspond to spatial patches
            patch_vec = attn_cls[-patch_count:]
            grid_size = int(math.sqrt(patch_count))
            if grid_size * grid_size != patch_count:
                grid_size = int(round(math.sqrt(patch_count)))

            patch_attn = patch_vec.reshape(grid_size, grid_size).cpu().numpy()

            # Normalize for visualization (0 to 1)
            patch_attn = (patch_attn - patch_attn.min()) / (patch_attn.max() - patch_attn.min() + 1e-8)

            # Resize to target heatmap size (must align with occlusion resolution)
            heatmap_resized = cv2.resize(patch_attn, self._heatmap_size[::-1])
            heatmaps.append(heatmap_resized.astype(np.float32))

        heatmaps_np = np.stack(heatmaps, axis=0)  # (B, H, W)
        heatmaps_tensor = torch.from_numpy(heatmaps_np)

        return self._normalize_attribution(heatmaps_tensor)

