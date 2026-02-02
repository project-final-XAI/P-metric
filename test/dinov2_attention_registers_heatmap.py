import argparse
import glob
import io
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms


# 1. Configuration & Setup
# ---------------------------------------------------------
DEFAULT_DATA_PATH = "../data/SIPaKMed_cropped"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")


# 2. File Loading Helper Functions
# ---------------------------------------------------------
def get_random_local_image(base_path: str) -> str:
    """
    Scans a directory recursively and returns a random image path.
    """
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"The path '{base_path}' does not exist on this machine.")

    print(f"Scanning {base_path} for images...")
    extensions = ("/*.jpg", "/*.jpeg", "/*.png", "/*.bmp", "/**/*.jpg", "/**/*.jpeg", "/**/*.png", "/**/*.bmp")
    all_files = []

    for ext in extensions:
        all_files.extend(glob.glob(base_path + ext, recursive=True))

    if not all_files:
        raise FileNotFoundError(f"No images found in {base_path}")

    selected = random.choice(all_files)
    print(f"Selected file: {selected}")
    return selected


def load_image(path_or_url: str) -> Image.Image:
    """
    Handles both HTTP URLs and local file paths.
    """
    if path_or_url.startswith("http"):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(path_or_url, headers=headers)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content)).convert("RGB")
    else:
        if not os.path.exists(path_or_url):
            raise FileNotFoundError(f"File not found: {path_or_url}")
        img = Image.open(path_or_url).convert("RGB")

    return img


# 3. Transformation Pipeline
# ---------------------------------------------------------
IMAGE_SIZE = (518, 518)

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


# 4. Attention Map Extraction
# ---------------------------------------------------------
def get_attention_map(model_name: str, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract a 2D attention map from a DINOv2 model loaded via torch.hub.

    Works for both standard and register variants:
      - "dinov2_vits14"
      - "dinov2_vits14_reg"
    """
    print(f"Loading model: {model_name}...")

    model = torch.hub.load("facebookresearch/dinov2", model_name, verbose=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Native attention extraction if available
        if hasattr(model, "get_last_selfattention"):
            attn = model.get_last_selfattention(img_tensor)
        else:
            # Manual fallback
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

    # Global attention (CLS token)
    attn_cls = attn[0, :, 0, :].mean(dim=0)

    # Reshape patch attention to 2D
    w, h = img_tensor.shape[2] // 14, img_tensor.shape[3] // 14
    num_patches = w * h
    attn_map = attn_cls[-num_patches:].reshape(w, h).cpu().numpy()

    # Normalize for visualization (0 to 1)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    return attn_map


def main():
    parser = argparse.ArgumentParser(
        description="DINOv2 attention heatmaps with optional register model comparison."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Root directory to search for images.",
    )
    parser.add_argument(
        "--use-registers",
        action="store_true",
        help="If set, also compute and show attention map from the register model.",
    )

    args = parser.parse_args()

    try:
        # --- STEP 1: Get Image ---
        # Try to get a local image first. If the folder doesn't exist, fallback to URL for testing.
        try:
            image_source = get_random_local_image(args.data_path)
            filename_display = os.path.basename(image_source)
        except Exception as e:
            print(f"Warning: Could not load local image ({e}). Using fallback URL.")
            image_source = (
                "https://upload.wikimedia.org/wikipedia/commons/9/99/"
                "Brooks_Chase_Ranger_of_Jolly_Dogs_Jack_Russell.jpg"
            )
            filename_display = "Web Image (Fallback)"

        # --- STEP 2: Process Image ---
        original_img_raw = load_image(image_source)
        vis_img = original_img_raw.resize(IMAGE_SIZE)
        img_tensor = transform(original_img_raw).unsqueeze(0).to(device)

        # --- STEP 3: Get Maps ---
        print("Calculating attention maps...")
        map_no_reg = get_attention_map("dinov2_vits14", img_tensor)

        map_with_reg = None
        if args.use_registers:
            map_with_reg = get_attention_map("dinov2_vits14_reg", img_tensor)

        # --- STEP 4: Visualization ---
        if map_with_reg is not None:
            # Show both standard and register maps + overlay of register
            fig, axes = plt.subplots(1, 4, figsize=(24, 6))
            fig.suptitle(
                f"File: {filename_display} (with registers comparison)",
                fontsize=14,
                fontweight="bold",
            )

            axes[0].imshow(vis_img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(map_no_reg, cmap="inferno")
            axes[1].set_title("DINOv2 Standard (Raw)")
            axes[1].axis("off")

            axes[2].imshow(map_with_reg, cmap="inferno")
            axes[2].set_title("DINOv2 Registers (Raw)")
            axes[2].axis("off")

            axes[3].imshow(vis_img)
            axes[3].imshow(
                map_with_reg,
                cmap="jet",
                alpha=0.5,
                interpolation="bilinear",
                extent=[0, IMAGE_SIZE[0], IMAGE_SIZE[1], 0],
            )
            axes[3].set_title("Registers Importance Overlay")
            axes[3].axis("off")
        else:
            # Only standard attention + overlay from standard
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(
                f"File: {filename_display} (no registers)",
                fontsize=14,
                fontweight="bold",
            )

            axes[0].imshow(vis_img)
            axes[0].set_title("Original Image")
            axes[0].axis("off")

            axes[1].imshow(map_no_reg, cmap="inferno")
            axes[1].set_title("DINOv2 Standard (Raw)")
            axes[1].axis("off")

            axes[2].imshow(vis_img)
            axes[2].imshow(
                map_no_reg,
                cmap="jet",
                alpha=0.5,
                interpolation="bilinear",
                extent=[0, IMAGE_SIZE[0], IMAGE_SIZE[1], 0],
            )
            axes[2].set_title("Standard Importance Overlay")
            axes[2].axis("off")

        print("\nDisplaying comparison...")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

