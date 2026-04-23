"""
test_u2net_random_image.py — Visual sanity check for U-2-Net saliency on random images.

Same idea as ``test/DINOv2/test_dino.py``: pick a random image from an ImageNet-style
folder tree, run the method, show Original | Heatmap | Overlay.

Run from repo root:
  python test/U2Net/test_u2net_random_image.py

Edit ``DATASET_KEY`` and/or ``BASE_PATH`` at the bottom if needed.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import types
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

import config
from data.loader import get_default_transforms

# ---------------------------------------------------------------------------
# Load ``unet_based`` without ``model_independent.__init__`` (sklearn / heavy DINO)
# ---------------------------------------------------------------------------
if "gdown" not in sys.modules:
    _gdown_stub = types.ModuleType("gdown")

    def _gdown_download(*_args, **_kwargs) -> None:
        raise RuntimeError("gdown stub: place models/u2net.pth or pip install gdown")

    _gdown_stub.download = _gdown_download  # type: ignore[attr-defined]
    sys.modules["gdown"] = _gdown_stub

_unet_path = _ROOT / "attribution" / "model_independent" / "unet_based.py"
_spec = importlib.util.spec_from_file_location(
    "attribution.model_independent.unet_based", _unet_path
)
_unet_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_unet_mod)

WEIGHTS_PATH = _unet_mod.WEIGHTS_PATH
U2NetSaliencyMethod = _unet_mod.U2NetSaliencyMethod


# ---------------------------------------------------------------------------
# Helpers (same pattern as test/DINOv2/test_dino.py)
# ---------------------------------------------------------------------------

def load_random_imagenet_img(base_path: str | Path):
    """Select and load a random image from an ImageNet-style directory tree."""
    base_path = Path(base_path)
    if not base_path.is_dir():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    subdirs = [d for d in base_path.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")

    random_dir = random.choice(subdirs)
    images = [
        f
        for f in os.listdir(random_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]

    if not images:
        return load_random_imagenet_img(base_path)

    random_img_name = random.choice(images)
    full_path = random_dir / random_img_name
    return Image.open(full_path).convert("RGB"), random_img_name


def run_u2net_on_random_images(base_path: str | Path, n_images: int = 2) -> None:
    if not WEIGHTS_PATH.is_file():
        print(f"Missing weights: {WEIGHTS_PATH}")
        print("Add u2net.pth under models/ or let the pipeline download it (gdown).")
        return

    transform = get_default_transforms()
    method = U2NetSaliencyMethod()
    device = torch.device(config.DEVICE)

    for idx in range(n_images):
        print(f"\n── Image {idx + 1}/{n_images} ──────────────────────────────────")
        try:
            img_pil, img_name = load_random_imagenet_img(base_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            continue

        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        print(f"[U2Net] computing for: {img_name}")

        with torch.inference_mode():
            hmap = method.compute(None, img_tensor, None)[0].cpu().numpy()

        H, W = hmap.shape
        img_np = np.array(img_pil.resize((W, H))) / 255.0

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"U-2-Net saliency — {img_name}", fontsize=14, weight="bold")

        titles = ["Original", "Heatmap", "Overlay"]
        for ax, title in zip(axes, titles):
            ax.set_title(title, fontsize=11)
            ax.axis("off")

        axes[0].imshow(img_np)

        im = axes[1].imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(img_np)
        im2 = axes[2].imshow(hmap, cmap="inferno", vmin=0.0, vmax=1.0, alpha=0.5)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Use config dataset path (same keys as config.DATASET_CONFIG)
    DATASET_KEY = "imagenet"  # e.g. "SIPaKMeD", "SIPaKMeD_cropped", "imagenet"
    cfg = getattr(config, "DATASET_CONFIG", {})
    if DATASET_KEY not in cfg:
        print(f"Unknown DATASET_KEY={DATASET_KEY!r}. Available: {list(cfg.keys())}")
        sys.exit(1)

    BASE_PATH = Path(cfg[DATASET_KEY]["path"])
    if not BASE_PATH.is_dir():
        print(f"Dataset path not found: {BASE_PATH}")
        print("Update DATASET_KEY or data paths in config.py.")
        sys.exit(1)

    run_u2net_on_random_images(BASE_PATH, n_images=3)
