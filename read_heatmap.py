import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def show_heatmap_with_images():
    # Relative paths
    heatmaps_path = os.path.join("results", "heatmaps")
    images_root = os.path.join("data", "imagenet")

    # Get all heatmap files
    heatmap_files = [f for f in os.listdir(heatmaps_path) if f.endswith(".npy")]
    selected_heatmaps = random.sample(heatmap_files, 5)

    # Get sorted list of imagenet folders
    imagenet_folders = sorted(
        [d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))]
    )

    for heatmap_file in selected_heatmaps:
        # Extract image number (last 5 digits)
        image_num_str = heatmap_file.split('_')[-1].split('.')[0]
        image_index = int(image_num_str)

        target_index = image_index

        # If index is beyond the number of folders, skip
        if target_index >= len(imagenet_folders):
            print(f"⚠️ No folder for index {target_index} (heatmap {heatmap_file})")
            continue

        # Pick the corresponding folder
        target_folder = imagenet_folders[target_index]
        folder_path = os.path.join(images_root, target_folder)

        # Find first image in that folder
        image_files = [f for f in os.listdir(folder_path)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print(f"⚠No images in folder {target_folder}")
            continue

        image_path = os.path.join(folder_path, image_files[0])
        class_name = image_path.split("_")[-1].split(".")[0]

        # Load the image and heatmap
        img = Image.open(image_path).convert("RGB")
        heatmap = np.load(os.path.join(heatmaps_path, heatmap_file))

        # Show side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img)
        axes[0].set_title(f"{class_name}")
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap='hot', interpolation='nearest')
        axes[1].set_title(f"Heatmap: {heatmap_file}")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    show_heatmap_with_images()