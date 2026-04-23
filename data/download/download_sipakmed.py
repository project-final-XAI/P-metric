import os
import shutil
import random
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("Error: Please install kagglehub first. Run: pip install kagglehub")
    exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Please install Pillow first. Run: pip install Pillow")
    exit(1)


def main():
    # Calculate target directory relative to this script
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent
    target_dir_test = project_root / "data" / "SIPaKMed_cropped"
    target_dir_train = project_root / "data" / "SIPaKMed_cropped_train"

    print("Downloading SIPaKMed dataset from Kaggle...")

    # Bypassing the buggy kagglehub MD5 checksum validation for large datasets
    import kagglehub.clients
    kagglehub.clients.get_md5_checksum_from_response = lambda response: None

    # Clear any corrupted/partial archives from cache before downloading.
    # A previous interrupted download leaves a .archive file that causes
    # "Bad magic number for file header" on the next run.
    cache_dir = (
            Path.home()
            / ".cache"
            / "kagglehub"
            / "datasets"
            / "prahladmehandiratta"
            / "cervical-cancer-largest-dataset-sipakmed"
    )
    if cache_dir.exists():
        for archive in cache_dir.rglob("*.archive"):
            print(f"Removing potentially corrupted archive: {archive}")
            archive.unlink()

    # This will return the path where the dataset was downloaded and cached.
    # If already extracted, it returns the cached path immediately.
    try:
        dataset_path = kagglehub.dataset_download(
            "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
        )
    except Exception as e:
        print(f"Download/extraction failed: {e}")
        print("Clearing full cache entry and retrying from scratch...")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        dataset_path = kagglehub.dataset_download(
            "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
        )

    print(f"Dataset downloaded to: {dataset_path}")

    dataset_path = Path(dataset_path)

    # Create the target directories if they don't exist
    target_dir_test.mkdir(parents=True, exist_ok=True)
    target_dir_train.mkdir(parents=True, exist_ok=True)

    print(f"Converting and copying cropped images...")
    print(f"Test (20%) will go to {target_dir_test}")
    print(f"Train (80%) will go to {target_dir_train}")

    # Search for all .bmp files
    bmp_files = list(dataset_path.rglob("*.bmp"))

    if not bmp_files:
        print("No .bmp files found in the downloaded dataset.")
        return

    # Group files by class
    class_files = {}
    for bmp_path in bmp_files:
        # Check if 'cropped' is in the path
        parts = bmp_path.parts

        if 'cropped' not in [p.lower() for p in parts]:
            continue

        # Determine the class name by looking for folders starting with 'im_'
        class_name = None
        for part in parts:
            if part.startswith("im_"):
                class_name = part
                break

        if not class_name:
            # Fallback if standard naming is not found
            class_name = "unknown_class"

        if class_name not in class_files:
            class_files[class_name] = []
        class_files[class_name].append(bmp_path)

    # Set a fixed seed for reproducible splits
    random.seed(42)

    count_train = 0
    count_test = 0

    for class_name, files in class_files.items():
        # Shuffle files to ensure random split
        random.shuffle(files)

        # Calculate split index
        split_idx = int(len(files) * 0.8)

        train_files = files[:split_idx]
        test_files = files[split_idx:]

        # Process train files
        train_class_dir = target_dir_train / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)

        for bmp_path in train_files:
            target_file = train_class_dir / (bmp_path.stem + ".jpeg")
            if not target_file.exists():
                try:
                    with Image.open(bmp_path) as img:
                        img = img.convert('RGB')
                        img.save(target_file, "JPEG", quality=100)
                    count_train += 1
                except Exception as e:
                    print(f"Error processing {bmp_path}: {e}")

        # Process test files
        test_class_dir = target_dir_test / class_name
        test_class_dir.mkdir(parents=True, exist_ok=True)

        for bmp_path in test_files:
            target_file = test_class_dir / (bmp_path.stem + ".jpeg")
            if not target_file.exists():
                try:
                    with Image.open(bmp_path) as img:
                        img = img.convert('RGB')
                        img.save(target_file, "JPEG", quality=100)
                    count_test += 1
                except Exception as e:
                    print(f"Error processing {bmp_path}: {e}")

        print(f"Processed class {class_name}: {len(train_files)} train, {len(test_files)} test")

    print(f"Done! Successfully processed {count_train} train images and {count_test} test images.")


if __name__ == "__main__":
    main()