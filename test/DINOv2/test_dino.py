"""
test_dino.py — Compare DINOv2 attribution methods side-by-side.

Run-all view includes:
  Original | Attn | PC1 | PC_EV | PC_L2 | ComboFixed | ComboEnt | ComboEnt Overlay

All heatmaps are continuous (non-binary) displays.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ── Import the new unified 6-method suite ────────────────────────────────────
# Adjust the import path if your project layout differs.
from attribution.dinov2_methods import (
    Dinov2AttnMethod,
    Dinov2ComboEntropyMethod,
    Dinov2ComboFixedMethod,
    Dinov2Pc1Method,
    Dinov2PcEigenweightedMethod,
    Dinov2PcL2Method,
)
from new_method import Dinov2UnifiedMethod, Dinov2PcaGaussianMethod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_random_imagenet_img(base_path: str):
    """Select and load a random image from an ImageNet-style directory tree."""
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base path {base_path} does not exist.")

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
    all 6 DINOv2 attribution methods side-by-side.
    """

    # 1. Instantiate all 6 methods
    # The internal caching in Dinov2Extractor ensures DINOv2 is only loaded once.
    methods = {
        # "Attn": Dinov2AttnMethod(),
        "PC1": Dinov2Pc1Method(),
        # "PC_EV": Dinov2PcEigenweightedMethod(),
        "PC_L2": Dinov2PcL2Method(),
        "ComboFixed": Dinov2ComboFixedMethod(),
        "ComboEnt": Dinov2ComboEntropyMethod(),
        "Gaussian": Dinov2PcaGaussianMethod(),
        "New_my" : Dinov2UnifiedMethod(),
    }

    for idx in range(n_images):
        print(f"\n── Image {idx + 1}/{n_images} ──────────────────────────────────")
        try:
            img_pil, img_name = load_random_imagenet_img(base_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        img_tensor = _transform(img_pil).unsqueeze(0)   # (1, 3, 518, 518)

        # 2. Compute heatmaps
        heatmaps = {}
        for name, method in methods.items():
            print(f"[{name:^10}] computing for: {img_name}")
            # [0] to grab the first item in the batch, then to numpy
            heatmaps[name] = method.compute(None, img_tensor, None)[0].cpu().numpy()

        # 3. Resize original to match heatmap spatial dims (for overlay)
        # Using the shape of the last computed heatmap
        H, W = list(heatmaps.values())[-1].shape
        img_np = np.array(img_pil.resize((W, H))) / 255.0

        # 4. Build figure
        # Columns: Original + 6 Methods + Overlay (ComboEnt)
        ncols = 2 + len(methods)
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        fig.suptitle(f"DINOv2 Attribution Suite — {img_name}", fontsize=14, weight='bold')

        # Setup column layout mapping
        plot_data = [("Original", None)] + list(heatmaps.items()) + [("ComboEnt Overlay", heatmaps["ComboEnt"])]

        for col, (ax, (title, hmap)) in enumerate(zip(axes, plot_data)):
            ax.set_title(title, fontsize=11)
            ax.axis("off")

            if hmap is None:
                # Original image column
                ax.imshow(img_np)
            elif col == ncols - 1:
                # Final overlay column (using the Combo Entropy map as the flagship)
                ax.imshow(img_np)
                im = ax.imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.5, interpolation="bilinear")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            else:
                # Pure heatmap columns
                im = ax.imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Adjust path as needed for your local environment
    imagenet_path = "../../data/imagenet"

    # Quick sanity check so it doesn't crash deep in matplotlib if the path is wrong
    if not os.path.exists(imagenet_path):
        print(f"⚠️ Warning: '{imagenet_path}' not found. Please update the path.")
    else:
        generate_comparison(imagenet_path, n_images=3)