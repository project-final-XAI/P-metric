import argparse
import os
import glob
import random
from io import BytesIO

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, Dinov2Model
import matplotlib.pyplot as plt


def get_random_local_image(base_path: str) -> str:
    """
    Recursively scans a directory and returns a random image path (jpg, jpeg, png).
    """
    extensions = ('/**/*.jpg', '/**/*.jpeg', '/**/*.png')
    all_files = []
    print(f"Scanning {base_path}...")
    for ext in extensions:
        all_files.extend(glob.glob(base_path + ext, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No images found in {base_path}")
    return random.choice(all_files)


def load_image(path_or_url: str) -> Image.Image:
    """
    Load an image from URL or local path and convert to RGB.
    """
    if path_or_url.startswith('http'):
        response = requests.get(path_or_url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"File not found: {path_or_url}")
        img = Image.open(path_or_url).convert('RGB')
    return img


def create_soft_heatmap(binary_mask: np.ndarray, sigma: float = 10) -> np.ndarray:
    """
    Turn a binary mask into a soft heatmap using distance transform + Gaussian blur.
    """
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    # Distance from background → high in object center
    dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    # Smooth to look like a soft attribution map
    heatmap = gaussian_filter(dist_transform, sigma=sigma)
    # Normalize to [0, 1]
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    return heatmap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinov2_model(use_registers: bool):
    """
    Load a DINOv2 model and processor.
    """
    if use_registers:
        # CORRECT ID for model with registers
        model_name = "facebook/dinov2-with-registers-small"
    else:
        # Standard model
        model_name = "facebook/dinov2-base"

    print(f"Loading model: {model_name} (registers={use_registers}) on {device}")

    # Use the Auto classes; they handle the architecture differences automatically
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Dinov2Model.from_pretrained(model_name).to(device)
    model.eval()
    return processor, model


def process_with_pca(image_path: str, use_registers: bool = False):
    img_orig = load_image(image_path)
    filename = os.path.basename(image_path)

    processor, model = load_dinov2_model(use_registers=use_registers)

    # 1. Prepare inputs and calculate Expected Grid
    inputs = processor(images=img_orig, return_tensors="pt").to(device)

    # DINOv2 usually resizes to 224, 518, etc. We must calculate the grid from actual tensor size.
    # inputs['pixel_values'] shape: (Batch, Channels, Height, Width)
    h = inputs['pixel_values'].shape[2]
    w = inputs['pixel_values'].shape[3]

    patch_size = 14
    grid_h = h // patch_size
    grid_w = w // patch_size
    num_patches_expected = grid_h * grid_w

    with torch.no_grad():
        outputs = model(**inputs)

        # 2. Dynamic Token Slicing (The Fix)
        # Get total sequence length from the model output
        # Shape: (Batch, Sequence_Length, Hidden_Size)
        seq_len = outputs.last_hidden_state.shape[1]

        # Calculate how many non-image tokens (CLS + Registers) are present
        num_extra_tokens = seq_len - num_patches_expected

        # Sanity check to prevent negative slicing
        if num_extra_tokens < 0:
            print(f"Error: Model output fewer tokens ({seq_len}) than expected patches ({num_patches_expected}).")
            return

        # Slice off whatever extra tokens exist (whether it's 1 or 5)
        features = outputs.last_hidden_state[:, num_extra_tokens:, :]

    features = features.squeeze(0).cpu().numpy()

    # 3. PCA and Visualization
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    # Reshape using the calculated grid dimensions
    try:
        obj_mask = pca_features[:, 0].reshape(grid_h, grid_w)
    except ValueError as e:
        print(f"Reshape Error: tried to reshape {features.shape[0]} into {grid_h}x{grid_w}")
        return

    # Normalize mask
    obj_mask = (obj_mask - obj_mask.min()) / (obj_mask.max() - obj_mask.min())

    # Invert heuristic
    if obj_mask[0, 0] > 0.5:
        obj_mask = 1 - obj_mask

    # Binary mask and heatmap
    binary_mask = (obj_mask > np.quantile(obj_mask, 0.85)).astype(float)
    binary_mask = cv2.resize(binary_mask, (img_orig.size[0], img_orig.size[1]))
    final_heatmap = create_soft_heatmap(binary_mask)

    show_results(img_orig, obj_mask, final_heatmap, filename, use_registers)

def show_results(img, pca_map, heatmap, title_name: str, use_registers: bool):
    """
    Show: original image, PCA map, and final soft heatmap.
    """
    fig = plt.figure(figsize=(15, 6))
    reg_tag = "with registers" if use_registers else "no registers"
    fig.suptitle(f"File: {title_name} ({reg_tag})", fontsize=14, fontweight='bold')

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pca_map, cmap='viridis')
    plt.title("PCA Component 1 (Object Isolation)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title("Final Soft Heatmap")
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 PCA + Gaussian heatmap visualization (with optional registers)."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data/imagenet",
        help="Root directory to search for images.",
    )
    parser.add_argument(
        "--use-registers",
        action="store_true",
        help="Use DINOv2 register model variant instead of standard.",
    )

    args = parser.parse_args()

    try:
        path = get_random_local_image(args.data_path)
        process_with_pca(path, use_registers=True)
    except Exception as e:
        print(f"Error encountered: {e}")


if __name__ == "__main__":
    main()

