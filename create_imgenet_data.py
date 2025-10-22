import os
import shutil

# github: https://github.com/EliSchwartz/imagenet-sample-images

# Path to the folder that currently contains all images
src_dir = r"C:\Users\1\Downloads\imagenet-sample-images-master\imagenet-sample-images-master"

# Path to the folder where you want the new structure
dst_dir = "data/imagenet"
os.makedirs(dst_dir, exist_ok=True)

# Loop through all files in the source directory
for filename in os.listdir(src_dir):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue  # Skip non-image files

    # Example filename: n01440764_tench.JPEG
    class_name = filename.split('_')[0]  # e.g., "n01440764"

    # Create a folder for the class if it doesn't exist
    class_folder = os.path.join(dst_dir, class_name)
    os.makedirs(class_folder, exist_ok=True)

    # Move the image into its class folder
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(class_folder, filename)
    shutil.move(src_path, dst_path)

print("Images reorganized into class folders.")
