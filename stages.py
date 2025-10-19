# stages.py

import os
from PIL import Image
import torch
from config import (
    MODEL_TYPE, MASK_MODE, GRADCAM_TARGET_LAYER,
    MASK_COLOR_TENSOR, IMAGENET_MEAN_TENSOR, device
)
from utils import preprocess, save_heatmap_jpg, save_tensor_img
import attribution_methods as am

# ============================================================
# STAGE 1: HEATMAP GENERATION
# ============================================================
def generate_heatmaps(image_path: str,
                      heatmap_folder: str,
                      model: torch.nn.Module,
                      index: int = None,
                      max_visuals: int = 10) -> str:
    """
    Compute and cache all heatmaps (.pt) for one image based on MODEL_TYPE.
    Saves preview JPGs for the first 'max_visuals' images.
    """
    os.makedirs(heatmap_folder, exist_ok=True)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # ---- Choose methods list ----
    cam_methods_cnn = ["gradcam", "guided_gradcam", "expected_gradcam"]
    cam_methods_vit = [] # CAM for ViT handled if needed, currently disabled in logic
    common_methods = [
        "occlusion", "naive_occ", "saliency", "inputxgradient", "guided_backprop",
        "smoothgrad", "integrated_gradients", "gradientshap", "xrai", "random_baseline",
    ]
    methods_list = cam_methods_cnn + common_methods if MODEL_TYPE == "cnn" else common_methods

    # ---- Check files to compute ----
    methods_to_compute = []
    for method_name in methods_list:
        method_dir   = os.path.join(heatmap_folder, method_name)
        heatmap_file = os.path.join(method_dir, f"{image_name}.pt")
        if not os.path.exists(heatmap_file):
            methods_to_compute.append(method_name)

    if not methods_to_compute:
        return f"SKIPPED: {image_name} (all heatmaps exist)"

    print(f"Processing {image_name}: computing {len(methods_to_compute)} methods {methods_to_compute}")

    # ---- Load & preprocess image ----
    image = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    # ---- Classify once (no-grad) to get target class ----
    with torch.no_grad():
        pred = model(img_tensor)
    class_idx = pred.argmax(dim=1).item()

    img_tensor = img_tensor.detach().requires_grad_(True)

    # ---- Build method table (name → callable) ----
    table = {
        "occlusion":            lambda: am.occlusion_mask(model, img_tensor, class_idx),
        "naive_occ":            lambda: am.naive_occ_mask(img_tensor),
        "saliency":             lambda: am.saliency_mask(model, img_tensor, class_idx),
        "inputxgradient":       lambda: am.inputxgradient_mask(model, img_tensor, class_idx),
        "guided_backprop":      lambda: am.guided_backprop_mask(model, img_tensor, class_idx),
        "smoothgrad":           lambda: am.smoothgrad_mask(model, img_tensor, class_idx),
        "integrated_gradients": lambda: am.integrated_gradients_mask(model, img_tensor, class_idx),
        "gradientshap":         lambda: am.gradientshap_mask(model, img_tensor, class_idx),
        "xrai":                 lambda: am.xrai_mask(model, img_tensor, class_idx),
        "random_baseline":      lambda: am.random_baseline_mask(img_tensor),
    }

    if MODEL_TYPE == "cnn":
        table.update({
            "gradcam":          lambda: am.gradcam_mask_once(model, img_tensor, class_idx),
            "guided_gradcam":   lambda: am.guided_gradcam_mask(model, img_tensor, class_idx),
            "expected_gradcam": lambda: am.expected_gradcam_mask(model, img_tensor, class_idx),
        })
    elif MODEL_TYPE == "vit":
         # Add ViT CAM here if desired
         pass

    # ---- Compute & save ----
    completed_methods = []
    for method_name in methods_to_compute:
        try:
            mask = table[method_name]()
            if mask is None: continue

            method_dir = os.path.join(heatmap_folder, method_name)
            os.makedirs(method_dir, exist_ok=True)
            torch.save(mask.cpu(), os.path.join(method_dir, f"{image_name}.pt"))

            if index is not None and index < max_visuals:
                save_heatmap_jpg(mask, os.path.join(method_dir, f"{image_name}.jpg"), original_image=image)

            completed_methods.append(method_name)

        except Exception as e:
            print(f"Error generating heatmap for {method_name}: {e}")
        finally:
            torch.cuda.empty_cache()

    return f"COMPLETED: {image_name} - {len(completed_methods)}/{len(methods_to_compute)} methods"

# ============================================================
# STAGE 2: OCCLUSION FROM HEATMAPS
# ============================================================
def mask_multiple_thresholds(img_tensor_norm, mask, thresholds):
    """
    Given one (1,1,H,W) heatmap, produce masked images for multiple quantile thresholds.
    """
    if img_tensor_norm.dim() == 3:
        img_tensor_norm = img_tensor_norm.unsqueeze(0)

    m      = mask.squeeze(0).squeeze(0)
    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)

    cutoffs = torch.quantile(m_norm.flatten(), torch.tensor(thresholds, device=m.device))

    results = {}
    for thr, cutoff in zip(thresholds, cutoffs):
        keep_mask    = (m_norm > cutoff).to(dtype=img_tensor_norm.dtype).unsqueeze(0).unsqueeze(0)
        keep_mask_3c = keep_mask.expand(img_tensor_norm.shape)

        if MASK_MODE == "random":
            rand_pixels_img  = torch.rand_like(img_tensor_norm)
            rand_pixels_norm = (rand_pixels_img - torch.zeros_like(rand_pixels_img)) / torch.ones_like(rand_pixels_img) # Simplified normalization for random noise
            masked_img_norm  = img_tensor_norm * keep_mask_3c + rand_pixels_norm * (1 - keep_mask_3c)
        elif MASK_MODE == "imagenet_mean":
            masked_img_norm  = img_tensor_norm * keep_mask_3c + IMAGENET_MEAN_TENSOR.expand_as(img_tensor_norm) * (1 - keep_mask_3c)
        else:  # "fixed"
            masked_img_norm  = img_tensor_norm * keep_mask_3c + MASK_COLOR_TENSOR.expand_as(img_tensor_norm) * (1 - keep_mask_3c)

        pct_removed = 100 * (1 - keep_mask.sum().item() / keep_mask.numel())
        results[thr] = (masked_img_norm, pct_removed)

    return results

def generate_occlusions_from_heatmaps(image_path, heatmap_folder, output_folder, thresholds):
    """
    For a single image: load each saved heatmap .pt, apply multiple thresholds,
    and save the masked images (JPG) in method-specific subfolders.
    """
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image      = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    for method_name in os.listdir(heatmap_folder):
        method_dir   = os.path.join(heatmap_folder, method_name)
        heatmap_path = os.path.join(method_dir, f"{image_name}.pt")
        if not os.path.exists(heatmap_path):
            continue

        mask = torch.load(heatmap_path, map_location=device).float()
        thr_results = mask_multiple_thresholds(img_tensor, mask, thresholds)

        for thr, (masked_tensor, pct_removed) in thr_results.items():
            out_dir = os.path.join(output_folder, method_name, f"{method_name}{int(thr * 100):02d}")
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"{image_name}_{method_name}{int(thr * 100):02d}.jpg")
            save_tensor_img(masked_tensor, save_path, pct_removed)