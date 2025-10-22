# utils/data_loader.py
"""
Handles dataset loading, preprocessing, and preparation.
Designed to be modular for easy extension to new datasets.
"""
import os
import shutil

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATASET_CONFIG


def get_default_transforms() -> transforms.Compose:
    """Provides standard ImageNet transforms for model compatibility."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def prepare_imagenet_subset(
        full_imagenet_val_path: str,
        subset_path: str,
        num_images_per_class: int = 1) -> None:
    """
    Creates a subset of the ImageNet validation set.

    It copies a specified number of images from each class folder
    to a new directory structure. This matches the paper's methodology
    [cite_start]of using one representative image per category. [cite: 236]

    Args:
        full_imagenet_val_path: Path to the full ImageNet validation set.
        subset_path: Path where the subset will be created.
        num_images_per_class: Number of images to copy from each class.
    """
    if os.path.exists(subset_path):
        print(f"Subset directory '{subset_path}' already exists. Skipping creation.")
        return

    print(f"Creating ImageNet subset at '{subset_path}'...")
    os.makedirs(subset_path, exist_ok=True)

    class_folders = sorted([d.name for d in os.scandir(full_imagenet_val_path) if d.is_dir()])

    for class_name in class_folders:
        source_class_dir = os.path.join(full_imagenet_val_path, class_name)
        target_class_dir = os.path.join(subset_path, class_name)
        os.makedirs(target_class_dir, exist_ok=True)

        images = sorted([f.name for f in os.scandir(source_class_dir) if f.is_file()])

        for i in range(min(num_images_per_class, len(images))):
            source_img_path = os.path.join(source_class_dir, images[i])
            target_img_path = os.path.join(target_class_dir, images[i])
            shutil.copy2(source_img_path, target_img_path)

    print("ImageNet subset creation complete.")


def get_dataloader(dataset_name: str, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """
    Creates and returns a DataLoader for a specified dataset.

    Args:
        dataset_name: The name of the dataset as defined in config.py.
        batch_size: The batch size for the DataLoader.
        shuffle: Whether to shuffle the data.

    Returns:
        A PyTorch DataLoader instance.

    Raises:
        ValueError: If the dataset_name is not found in the config.
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_CONFIG.")

    dataset_path = DATASET_CONFIG[dataset_name]["path"]

    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' is empty or does not exist. "
            f"Please run `prepare_imagenet_subset` or place your data there."
        )

    image_folder = datasets.ImageFolder(
        root=dataset_path,
        transform=get_default_transforms()
    )

    dataloader = DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2
    )

    return dataloader
