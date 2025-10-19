# utils.py

import os
import numpy as np
from PIL import Image
import torch
import matplotlib.cm as cm
from torchvision import transforms
from config import device, MEAN, STD, DEFAULT_MASK_COLOR

# ============================================================
# DEVICE & NORMALIZATION UTILS
# ============================================================
def get_mask_color_tensor(mask_color):
    """Convert an RGB mask color into normalized tensor space so compositing is consistent."""
    color = torch.tensor(mask_color, device=device).view(1, 3, 1, 1)
    return (color - MEAN) / STD

MASK_COLOR_TENSOR    = get_mask_color_tensor(DEFAULT_MASK_COLOR)
IMAGENET_MEAN_TENSOR = (MEAN - MEAN) / STD  # zeros in normalized space

# Standard ImageNet preprocessing for 224×224 models
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize(t: torch.Tensor) -> torch.Tensor:
    """Undo normalization back to [0,1] space (clipping happens at save)."""
    return t * STD + MEAN

# ============================================================
# MODULE UTILS
# ============================================================
def get_module_by_name(model: torch.nn.Module, dotted: str):
    """Resolve a nested module by its dotted path (e.g., 'encoder.layers.encoder_layer_11.ln_1')."""
    cur = model
    for p in dotted.split('.'):
        if not hasattr(cur, p):
            return None
        cur = getattr(cur, p)
    return cur

def _get_target_module(model: torch.nn.Module, dotted: str):
    """Utility to safely retrieve a module by dotted path."""
    cur = model
    for p in dotted.split('.'):
        if not hasattr(cur, p):
            raise AttributeError(f"Target module '{dotted}' not found (missing '{p}')")
        cur = getattr(cur, p)
    return cur

# ============================================================
# VISUALIZATION HELPERS
# ============================================================
def normalize_attribution_for_display(attr_tensor: torch.Tensor, percentile_clip: float = 1.0) -> torch.Tensor:
    """Normalize an attribution map for *display only*."""
    a = attr_tensor.detach().float().cpu()
    if a.ndim == 4:
        a = a.squeeze(0).squeeze(0)
    elif a.ndim == 3:
        a = a.squeeze(0)

    a = a.abs()
    if 0 < percentile_clip < 50:
        low  = torch.quantile(a, percentile_clip / 100.0)
        high = torch.quantile(a, 1.0 - percentile_clip / 100.0)
        a = a.clamp(low, high)

    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a  # (H, W) in [0,1]

def save_tensor_img(tensor: torch.Tensor, path: str, pct_removed=None):
    """Save a denormalized tensor as a JPG. Skips if file exists to save time."""
    if os.path.exists(path):
        return
    img = denormalize(tensor.detach()).squeeze(0).clamp(0, 1)
    img_np = img.permute(1, 2, 0).cpu().numpy()
    Image.fromarray((img_np * 255).astype(np.uint8)).save(path, quality=95)

def create_overlay(original_image: Image.Image, colored_heatmap: np.ndarray, alpha: float = 0.6) -> Image.Image:
    """Blend heatmap with original image for visual inspection."""
    orig_resized = original_image.resize((colored_heatmap.shape[1], colored_heatmap.shape[0]))
    orig_np = np.array(orig_resized).astype(np.float32) / 255.0
    overlay_np = (1 - alpha) * orig_np + alpha * colored_heatmap
    overlay_np = np.clip(overlay_np, 0, 1)
    return Image.fromarray((overlay_np * 255).astype(np.uint8))

def save_heatmap_jpg(mask_tensor: torch.Tensor, path: str, original_image: Image.Image = None):
    """
    Save a colored heatmap (.jpg). If original_image provided, also save an overlay image.
    Uses percentile-based normalization for nicer contrast across methods.
    """
    heatmap_norm = normalize_attribution_for_display(mask_tensor, percentile_clip=1.0).numpy()
    colored = cm.viridis(heatmap_norm)[:, :, :3]
    Image.fromarray((colored * 255).astype(np.uint8)).save(path, quality=95)

    if original_image is not None:
        overlay = create_overlay(original_image, colored, alpha=0.6)
        overlay_path = path.replace(".jpg", "_overlay.jpg")
        overlay.save(overlay_path, quality=95)