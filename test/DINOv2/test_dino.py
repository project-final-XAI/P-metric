"""
test_dino.py — Compare DINOv2 attribution methods side-by-side.

Run-all view includes:
  Original | PCA-Gaussian | CLS-Attention | sumDino | pca1 | Unified | Unified overlay

All heatmaps are continuous (non-binary) displays.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── new combined method ──────────────────────────────────────────────────────
from gussian import Dinov2UnifiedMethod

# ── original individual methods ──────────────────────────────────────────────
# Adjust the import paths if your project layout differs.
try:
    from attribution.dinov2_methods import (
        Dinov2PcaGaussianMethod,
        Dinov2AttentionMethod,
        SumDinoMethod,
        Dinov2Pca1Method,
    )
    _HAVE_OLD_METHODS = True
except ImportError:
    print(
        "[test_dino] Could not import old methods "
        "(attribution.method_dinov2 not found). "
        "Only the fused method will be shown."
    )
    _HAVE_OLD_METHODS = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_random_imagenet_img(base_path: str):
    """Select and load a random image from an ImageNet-style directory tree."""
    subdirs = [
        d for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    random_dir = random.choice(subdirs)
    dir_path   = os.path.join(base_path, random_dir)
    images     = [
        f for f in os.listdir(dir_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not images:
        return load_random_imagenet_img(base_path)   # try another dir

    random_img_name = random.choice(images)
    full_path       = os.path.join(dir_path, random_img_name)
    return Image.open(full_path).convert("RGB"), random_img_name


_transform = transforms.Compose([
    transforms.Resize((518, 518)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_comparison(base_path: str, n_images: int = 2) -> None:
    """
    Loads ``n_images`` random images and produces a comparison figure showing
    all available methods side-by-side.
    """
    # Instantiate methods once (models are cached internally)
    fused_method = Dinov2UnifiedMethod()
    if _HAVE_OLD_METHODS:
        pca_method  = Dinov2PcaGaussianMethod()
        attn_method = Dinov2AttentionMethod()
        sum_method  = SumDinoMethod()
        pca1_method = Dinov2Pca1Method()

    for idx in range(n_images):
        print(f"\n── Image {idx + 1}/{n_images} ──────────────────────────────────")
        try:
            img_pil, img_name = load_random_imagenet_img(base_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        img_tensor = _transform(img_pil).unsqueeze(0)   # (1, 3, 518, 518)

        # ── compute heatmaps ─────────────────────────────────────────────────
        print(f"[fused]   computing for: {img_name}")
        fused_heatmap = fused_method.compute(None, img_tensor, None)[0].cpu().numpy()

        pca_heatmap = attn_heatmap = sum_heatmap = pca1_heatmap = None
        if _HAVE_OLD_METHODS:
            print(f"[pca]     computing for: {img_name}")
            pca_heatmap  = pca_method.compute(None, img_tensor, None)[0].cpu().numpy()
            print(f"[attn]    computing for: {img_name}")
            attn_heatmap = attn_method.compute(None, img_tensor, None)[0].cpu().numpy()
            print(f"[sumDino] computing for: {img_name}")
            sum_heatmap  = sum_method.compute(None, img_tensor, None)[0].cpu().numpy()
            print(f"[pca1]    computing for: {img_name}")
            pca1_heatmap = pca1_method.compute(None, img_tensor, None)[0].cpu().numpy()

        # ── resize original to match heatmap spatial dims ───────────────────
        H, W    = fused_heatmap.shape
        img_np  = np.array(img_pil.resize((W, H))) / 255.0

        # ── build figure ─────────────────────────────────────────────────────
        if _HAVE_OLD_METHODS:
            ncols = 7
            titles = [
                f"Original\n{img_name}",
                "PCA-Gaussian\n(old)",
                "CLS-Attention\n(old)",
                "sumDino\n(attn + pca)",
                "pca1\n(PC1 only)",
                "Unified\n(new)",
                "Unified Overlay\n(new)",
            ]
            maps = [None, pca_heatmap, attn_heatmap, sum_heatmap, pca1_heatmap, fused_heatmap, fused_heatmap]
        else:
            ncols = 4
            titles = [
                f"Original\n{img_name}",
                "Fused heatmap",
                "Fused (jet)",
                "Fused Overlay",
            ]
            maps = [None, fused_heatmap, fused_heatmap, fused_heatmap]

        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        fig.suptitle(f"DINOv2 Attribution Comparison — {img_name}", fontsize=13)

        for col, (ax, title, hmap) in enumerate(zip(axes, titles, maps)):
            ax.set_title(title, fontsize=9)
            ax.axis("off")

            if hmap is None:
                # Original image
                ax.imshow(img_np)
            elif col == ncols - 1:
                # Overlay column
                ax.imshow(img_np)
                im = ax.imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.45, interpolation="bilinear")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Pure heatmap column
                im = ax.imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    imagenet_path = "../../data/imagenet"   # data/imagenet (repo-relative)
    generate_comparison(imagenet_path, n_images=2)