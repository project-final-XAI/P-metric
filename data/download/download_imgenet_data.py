import os
import shutil
import urllib.request
import zipfile
import tempfile
from pathlib import Path

def main():
    url = "https://github.com/EliSchwartz/imagenet-sample-images/archive/refs/heads/master.zip"
    
    # Calculate target directory relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    dst_dir = project_root / "data" / "imagenet"
    
    print("Downloading imagenet-sample-images from GitHub...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "imagenet_samples.zip")
        
        try:
            urllib.request.urlretrieve(url, zip_path)
        except Exception as e:
            print(f"Error downloading the dataset: {e}")
            return
            
        print("Extracting images...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            
        # The zip contains a folder named 'imagenet-sample-images-master'
        src_dir = os.path.join(temp_dir, "imagenet-sample-images-master")
        
        if not os.path.exists(src_dir):
            print("Extracted folder structure is different than expected.")
            return
            
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Organizing images into {dst_dir}...")
        
        count = 0
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
            
            # Move the file
            shutil.move(src_path, dst_path)
            count += 1
            
        print(f"Successfully downloaded and reorganized {count} images into class folders.")

if __name__ == "__main__":
    main()
