import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import argparse

def show_heatmap_with_images(dataset_name='imagenet', num_samples=5):
    """
    Display random heatmaps with their corresponding images.
    
    Args:
        dataset_name: Name of the dataset ('imagenet' or 'SIPaKMeD')
        num_samples: Number of random samples to display
    """
    # Relative paths
    heatmaps_path = Path("results") / "heatmaps" / dataset_name
    images_root = Path("data") / dataset_name
    
    if not heatmaps_path.exists():
        print(f"Heatmap directory not found: {heatmaps_path}")
        return
    
    if not images_root.exists():
        print(f"Image directory not found: {images_root}")
        return

    # Get all heatmap files (non-sorted)
    heatmap_files = [
        f for f in os.listdir(heatmaps_path)
        if f.endswith(".npy") and not f.endswith("_sorted.npy")
    ]
    
    if not heatmap_files:
        print(f"No heatmap files found in {heatmaps_path}")
        return
    
    num_to_show = min(num_samples, len(heatmap_files))
    selected_heatmaps = random.sample(heatmap_files, num_to_show)

    # Get sorted list of class folders
    class_folders = sorted([
        d for d in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, d))
    ])
    
    print(f"Found {len(class_folders)} classes in {dataset_name}")
    print(f"Showing {num_to_show} random heatmaps")

    for heatmap_file in selected_heatmaps:
        try:
            # Extract image number from filename (format: model-method-image_XXXXX.npy)
            image_id = heatmap_file.split('-')[-1].split('.')[0]  # e.g., "image_00001"
            image_num_str = image_id.split('_')[-1]  # e.g., "00001"
            image_index = int(image_num_str)

            # If index is beyond the number of folders, skip
            if image_index >= len(class_folders):
                print(f"No folder for index {image_index} (heatmap {heatmap_file})")
                continue

            # Pick the corresponding folder
            target_folder = class_folders[image_index]
            folder_path = os.path.join(images_root, target_folder)

            # Find first image in that folder
            image_files = [
                f for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not image_files:
                print(f"No images in folder {target_folder}")
                continue

            image_path = os.path.join(folder_path, image_files[0])
            
            # Load the image and heatmap
            img = Image.open(image_path).convert("RGB")
            heatmap = np.load(heatmaps_path / heatmap_file)

            # Show side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            axes[0].imshow(img)
            axes[0].set_title(f"Class: {target_folder}")
            axes[0].axis("off")

            axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
            axes[1].set_title(f"Heatmap\n{heatmap_file}")
            axes[1].axis("off")

            plt.suptitle(f"Dataset: {dataset_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error processing {heatmap_file}: {e}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize heatmaps with images")
    parser.add_argument(
        '--dataset', 
        default='imagenet',
        help='Dataset name (imagenet, SIPaKMeD)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Number of random samples to display'
    )
    args = parser.parse_args()
    
    show_heatmap_with_images(args.dataset, args.num_samples)