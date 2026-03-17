import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os
import random
from gussian import Dinov2BinarySegmentMethod

def load_random_imagenet_img(base_path: str):
    """Selects and loads a random image from ImageNet style directory structure."""
    # 1. Get list of all subdirectories (synsets)
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories found in {base_path}")
    
    # 2. Pick a random subdirectory
    random_dir = random.choice(subdirs)
    dir_path = os.path.join(base_path, random_dir)
    
    # 3. Pick a random image from that directory
    images = [f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        # If directory is empty, try again recursively or skip (simplified here)
        return load_random_imagenet_img(base_path)
        
    random_img_name = random.choice(images)
    full_path = os.path.join(dir_path, random_img_name)
    
    print(f"Selected Image: {full_path}")
    return Image.open(full_path).convert("RGB")

def generate_heatmap_local(base_path: str):
    # Load local random image instead of URL
    try:
        img_pil = load_random_imagenet_img(base_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    img_tensor = transform(img_pil).unsqueeze(0)

    # Initialize Method
    pca_tool = Dinov2BinarySegmentMethod()

    # Compute
    print("Generating PCA heatmap...")
    # Get heatmap and ensure it's a numpy array on CPU
    pca_heatmap = pca_tool.compute(None, img_tensor, None)[0].cpu().numpy()

    # --- Overlay Logic ---
    img_np = np.array(img_pil) / 255.0
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(img_pil)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(pca_heatmap, cmap='jet')
    ax[1].set_title("DINOv2 PCA Heatmap")
    ax[1].axis("off")

    ax[2].imshow(img_np)
    ax[2].imshow(pca_heatmap, cmap='jet', alpha=0.5)
    ax[2].set_title("Overlap (Blend)")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Path relative to your test_dino.py file
    imagenet_path = "../../data/imagenet" 
    generate_heatmap_local(imagenet_path)