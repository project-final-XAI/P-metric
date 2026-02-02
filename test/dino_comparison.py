import argparse
import os
import random
import glob
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms

from dinov2_pca_gaussian_heatmap import (
    load_dinov2_model as load_dinov2_pca_model,
    create_soft_heatmap,
)
from transformers import AutoImageProcessor, Dinov2Model
from sklearn.decomposition import PCA
import cv2


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_random_local_image(base_path: str) -> str:
    exts = ("/*.jpg", "/*.jpeg", "/*.png", "/**/*.jpg", "/**/*.jpeg", "/**/*.png")
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(base_path + ext, recursive=True))
    if not all_files:
        raise FileNotFoundError(f"No images found in {base_path}")
    return random.choice(all_files)


def load_image(path_or_url: str) -> Image.Image:
    if path_or_url.startswith("http"):
        resp = requests.get(path_or_url)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"File not found: {path_or_url}")
    return Image.open(path_or_url).convert("RGB")


IMAGE_SIZE = (224, 224)
ATTN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


def compute_pca_heatmap(img_pil: Image.Image, use_registers: bool) -> np.ndarray:
    processor, model = load_dinov2_pca_model(use_registers=use_registers)

    inputs = processor(images=img_pil, return_tensors="pt").to(DEVICE)

    h = inputs["pixel_values"].shape[2]
    w = inputs["pixel_values"].shape[3]
    patch_size = 14
    grid_h = h // patch_size
    grid_w = w // patch_size
    num_patches_expected = grid_h * grid_w

    with torch.no_grad():
        outputs = model(**inputs)
        seq_len = outputs.last_hidden_state.shape[1]
        num_extra_tokens = seq_len - num_patches_expected
        if num_extra_tokens < 0:
            raise RuntimeError(
                f"Unexpected token layout: seq_len={seq_len}, expected_patches={num_patches_expected}"
            )
        features = outputs.last_hidden_state[:, num_extra_tokens:, :]

    features = features.squeeze(0).cpu().numpy()
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(features)

    obj_mask = pca_features[:, 0].reshape(grid_h, grid_w)
    obj_mask = (obj_mask - obj_mask.min()) / (obj_mask.max() - obj_mask.min() + 1e-8)
    if obj_mask[0, 0] > 0.5:
        obj_mask = 1 - obj_mask

    binary_mask = (obj_mask > np.quantile(obj_mask, 0.85)).astype(float)
    binary_mask = cv2.resize(binary_mask, (img_pil.size[0], img_pil.size[1]))
    final_heatmap = create_soft_heatmap(binary_mask)

    return final_heatmap


def compute_attention_heatmap(img_pil: Image.Image, use_registers: bool) -> np.ndarray:
    model_name = "dinov2_vits14_reg" if use_registers else "dinov2_vits14"
    print(f"Loading attention model: {model_name} on {DEVICE}")
    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.to(DEVICE)
    model.eval()

    img_tensor = ATTN_TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        if hasattr(model, "get_last_selfattention"):
            attn = model.get_last_selfattention(img_tensor)
        else:
            x = model.prepare_tokens_with_masks(img_tensor)
            for i, blk in enumerate(model.blocks):
                if i < len(model.blocks) - 1:
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

    attn_cls = attn[0, :, 0, :].mean(dim=0)

    # derive patch tokens (exclude CLS and any registers)
    num_tokens = attn_cls.shape[0]
    # assume 16x16 grid
    num_patches = 16 * 16
    if num_patches > num_tokens - 1:
        num_patches = num_tokens - 1
    patch_vec = attn_cls[-num_patches:]
    grid_size = int(np.sqrt(num_patches))
    attn_map = patch_vec.reshape(grid_size, grid_size).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_map = cv2.resize(attn_map, (img_pil.size[0], img_pil.size[1]))
    return attn_map


def main():
    parser = argparse.ArgumentParser(
        description="Compare DINOv2 PCA+Gaussian heatmap vs attention heatmap (with optional registers)."
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
        help="Use register variants for both PCA and attention where applicable.",
    )
    args = parser.parse_args()

    img_path = get_random_local_image(args.data_path)
    img_pil = load_image(img_path)

    pca_heatmap = compute_pca_heatmap(img_pil, use_registers=args.use_registers)
    attn_heatmap = compute_attention_heatmap(img_pil, use_registers=args.use_registers)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    title_tag = "with registers" if args.use_registers else "no registers"
    fig.suptitle(f"File: {os.path.basename(img_path)} ({title_tag})", fontsize=14, fontweight="bold")

    axes[0].imshow(img_pil)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(img_pil)
    axes[1].imshow(pca_heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title("DINOv2 PCA + Gaussian")
    axes[1].axis("off")

    axes[2].imshow(img_pil)
    axes[2].imshow(attn_heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title("DINOv2 Attention")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
