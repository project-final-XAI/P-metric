"""
Data loading utilities.

Handles dataset loading, preprocessing, and preparation.
"""

import os
import logging
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATASET_CONFIG


def get_default_transforms() -> transforms.Compose:
    """Standard ImageNet transforms for model compatibility."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloader(dataset_name: str, batch_size: int = 1, shuffle: bool = False) -> DataLoader:
    """
    Create DataLoader for specified dataset.
    
    Args:
        dataset_name: Dataset name from config
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
        
    Returns:
        PyTorch DataLoader instance
        
    Raises:
        ValueError: If dataset not found in config
        FileNotFoundError: If dataset directory doesn't exist
    """
    if dataset_name not in DATASET_CONFIG:
        raise ValueError(f"Dataset '{dataset_name}' not found in DATASET_CONFIG.")

    dataset_path = DATASET_CONFIG[dataset_name]["path"]

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' does not exist. "
            f"Please prepare your data first."
        )
    
    if not os.listdir(dataset_path):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_path}' is empty. "
            f"Please add images to the directory."
        )

    image_folder = datasets.ImageFolder(
        root=dataset_path,
        transform=get_default_transforms()
    )
    
    logging.info(f"Found {len(image_folder)} images in {dataset_path}")

    # Auto-adjust workers and pin_memory based on device
    import torch
    num_workers = 4 if torch.cuda.is_available() else 2
    pin_memory = torch.cuda.is_available()

    dataloader = DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader

