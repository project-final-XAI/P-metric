"""
Data loading utilities.

Handles dataset loading, preprocessing, and preparation.
Also contains dataset handler interfaces and factory logic.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATASET_CONFIG, MAX_WORKERS


def get_base_transforms() -> transforms.Compose:
    """
    Minimal base transforms.
    Resizing is now handled dynamically by the AttributionMethod base class
    using the `target_size` parameter.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# --- 1. Generic Hardware-Optimized Loader ---

def create_dataloader(dataset_path: str, transforms_pipeline: transforms.Compose, batch_size: int, shuffle: bool) -> DataLoader:
    """Core PyTorch loader that handles GPU optimization."""
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' is missing or empty.")

    image_folder = datasets.ImageFolder(root=dataset_path, transform=transforms_pipeline)
    logging.info(f"Found {len(image_folder)} images in {dataset_path}")

    # Optimize for hardware
    if torch.cuda.is_available():
        return DataLoader(
            dataset=image_folder,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=MAX_WORKERS,
            pin_memory=True,
            persistent_workers=(len(image_folder) > 10),
            prefetch_factor=4
        )
    else:
        return DataLoader(
            dataset=image_folder,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=2
        )


# --- 2. Base Interface ---

class BaseDatasetHandler(ABC):
    """Defines the contract for all dataset handlers."""

    @abstractmethod
    def get_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        """Return a DataLoader for this dataset."""
        pass

    @abstractmethod
    def get_category_name(self, label: int) -> Optional[str]:
        """Map class label to a human-readable name."""
        pass

    @abstractmethod
    def get_synset_id(self, label: int) -> str:
        """Return the physical folder name (e.g. Synset ID) for a label."""
        pass


# --- 3. ImageNet Handler ---

class ImageNetHandler(BaseDatasetHandler):
    """Handler for the ImageNet dataset."""

    def __init__(self, dataset_name: str = "imagenet") -> None:
        self._dataset_name = dataset_name
        self.dataset_path = DATASET_CONFIG.get(dataset_name, {}).get("path", "")
        self._dataset = None  # Cache the dataset to access folder names later

    def get_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        loader = create_dataloader(self.dataset_path, get_base_transforms(), batch_size, shuffle)
        self._dataset = loader.dataset  # Store reference to the ImageFolder
        return loader

    def get_category_name(self, label: int) -> Optional[str]:
        # Fast, O(1) lookup using our new utility! No more file scanning.
        from data.imagenet_class_mapping import get_imagenet_category_name
        return get_imagenet_category_name(label)

    def get_synset_id(self, label: int) -> str:
        # Ask PyTorch's ImageFolder directly for the folder name
        if self._dataset is not None:
            return self._dataset.classes[label]
        return f"unknown_synset_{label}"


# --- 4. SIPaKMeD Handler ---

class SIPaKMeDHandler(BaseDatasetHandler):
    """Handler for the SIPaKMeD dataset."""

    _CLASSES = {
        0: "Dyskeratotic",
        1: "Koilocytotic",
        2: "Metaplastic",
        3: "Parabasal",
        4: "Superficial-Intermediate",
    }

    def __init__(self, dataset_name: str = "SIPaKMeD_cropped") -> None:
        self._dataset_name = dataset_name
        self.dataset_path = DATASET_CONFIG.get(dataset_name, {}).get("path", "")
        self._dataset = None

    def get_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        loader = create_dataloader(self.dataset_path, get_base_transforms(), batch_size, shuffle)
        self._dataset = loader.dataset
        return loader

    def get_category_name(self, label: int) -> Optional[str]:
        return self._CLASSES.get(label, f"class_{label}")

    def get_synset_id(self, label: int) -> str:
        # Will return exactly the folder name in your SIPaKMeD dataset directory
        if self._dataset is not None:
            return self._dataset.classes[label]
        return f"class_{label}"


# --- 5. Factory ---

def get_dataset_handler(dataset_name: str) -> BaseDatasetHandler:
    """Factory: return the appropriate dataset handler."""
    if dataset_name == "imagenet":
        return ImageNetHandler(dataset_name)
    elif dataset_name in ("SIPaKMeD", "SIPaKMeD_cropped"):
        return SIPaKMeDHandler(dataset_name)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: imagenet, SIPaKMeD, SIPaKMeD_cropped"
        )


if __name__ == "__main__":
    import time


    def test_dataset(dataset_name: str):
        print(f"\n" + "=" * 50)
        print(f" TESTING DATASET: {dataset_name}")
        print("=" * 50)

        try:
            # 1. Initialize the handler via the Factory
            handler = get_dataset_handler(dataset_name)
            print(f"[✓] Successfully initialized {type(handler).__name__}")

            # 2. Get the DataLoader
            batch_size = 1
            dataloader = handler.get_dataloader(batch_size=batch_size, shuffle=True)

            # 3. Fetch a single batch
            images, labels = next(iter(dataloader))

            # 4. Test the mapping utilities
            for i, label_idx in enumerate(labels.tolist()):
                human_name = handler.get_category_name(label_idx)
                folder_name = handler.get_synset_id(label_idx)

                print(f"    Sample {i + 1}:")
                print(f"      - Integer Label: {label_idx}")
                print(f"      - Folder/Synset: {folder_name}")
                print(f"      - Human Name:    {human_name}")

        except Exception as e:
            print(f"[X] Error testing {dataset_name}: {e}")


    # Set up basic logging so we can see the create_dataloader info prints
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Test SIPaKMeD (Requires valid path in config.py)
    test_dataset("SIPaKMeD_cropped")

    # Test ImageNet (Requires valid path in config.py)
    test_dataset("imagenet")