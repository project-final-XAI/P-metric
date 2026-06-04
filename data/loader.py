"""
Data loading utilities.

Handles dataset loading, preprocessing, and preparation.
Also contains dataset handler interfaces and factory logic.
"""

from abc import ABC, abstractmethod
import logging
import os
from typing import Any, Optional
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import DATASET_CONFIG, MAX_WORKERS


def get_base_transforms() -> transforms.Compose:
    """
    Standard transforms based on dataset.
    Uses ImageNet standard validation pipeline (Resize 256 -> CenterCrop 224)
    for ImageNet, and direct Resize 224 for SIPaKMeD.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DualImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder that returns both the classification transform,
    the attribution transform, and the original path.
    """
    def __init__(self, root, clf_transform, attr_transform, **kwargs):
        super().__init__(root, **kwargs)
        self.clf_transform = clf_transform
        self.attr_transform = attr_transform

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        
        clf_tensor = self.clf_transform(sample)
        attr_tensor = self.attr_transform(sample)
        
        return (clf_tensor, attr_tensor, path), target


def get_clf_transform(dataset_name: str) -> transforms.Compose:
    if dataset_name == "imagenet":
        return transforms.Compose([
            transforms.Resize(232, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def get_attr_transform(dataset_name: str) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# --- 1. Generic Hardware-Optimized Loader ---

def create_dataloader(
    dataset_path: str,
    transforms_pipeline: Optional[transforms.Compose],
    batch_size: int,
    shuffle: bool,
    clf_transform: Optional[transforms.Compose] = None,
    attr_transform: Optional[transforms.Compose] = None
) -> DataLoader:
    """Core PyTorch loader that handles GPU optimization."""
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        raise FileNotFoundError(f"Dataset directory '{dataset_path}' is missing or empty.")

    if clf_transform is not None and attr_transform is not None:
        image_folder = DualImageFolder(root=dataset_path, clf_transform=clf_transform, attr_transform=attr_transform)
    else:
        image_folder = datasets.ImageFolder(root=dataset_path, transform=transforms_pipeline)
    logging.info(f"Found {len(image_folder)} images in {dataset_path}")

    # Optimize for hardware
    if torch.cuda.is_available():
        num_workers = MAX_WORKERS
        if num_workers > 0:
            return DataLoader(
                dataset=image_folder,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=(len(image_folder) > 10),
                prefetch_factor=4
            )
        else:
            return DataLoader(
                dataset=image_folder,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0,
                pin_memory=True,
                persistent_workers=False,
                prefetch_factor=None
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
    def get_dataloader(self, batch_size: int, shuffle: bool = False, transform: Optional[Any] = None) -> DataLoader:
        """Return a DataLoader for this dataset."""
        pass

    @abstractmethod
    def get_dual_dataloader(self, batch_size: int, shuffle: bool = False, clf_transform: Optional[Any] = None) -> DataLoader:
        """Return a DataLoader that yields dual tensors (clf, attr, path) for Phase 1."""
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

    def get_dataloader(self, batch_size: int, shuffle: bool = False, transform: Optional[Any] = None) -> DataLoader:
        if transform is None:
            transform = get_base_transforms()
        loader = create_dataloader(self.dataset_path, transform, batch_size, shuffle)
        self._dataset = loader.dataset  # Store reference to the ImageFolder
        return loader

    def get_dual_dataloader(self, batch_size: int, shuffle: bool = False, clf_transform: Optional[Any] = None) -> DataLoader:
        if clf_transform is None:
            clf_transform = get_clf_transform(self._dataset_name)
        attr_transform = get_attr_transform(self._dataset_name)
        loader = create_dataloader(
            self.dataset_path,
            None,
            batch_size,
            shuffle,
            clf_transform=clf_transform,
            attr_transform=attr_transform
        )
        self._dataset = loader.dataset
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

    def get_dataloader(self, batch_size: int, shuffle: bool = False, transform: Optional[Any] = None) -> DataLoader:
        if transform is None:
            transform = get_base_transforms()
        loader = create_dataloader(self.dataset_path, transform, batch_size, shuffle)
        self._dataset = loader.dataset
        return loader

    def get_dual_dataloader(self, batch_size: int, shuffle: bool = False, clf_transform: Optional[Any] = None) -> DataLoader:
        if clf_transform is None:
            clf_transform = get_clf_transform(self._dataset_name)
        attr_transform = get_attr_transform(self._dataset_name)
        loader = create_dataloader(
            self.dataset_path,
            None,
            batch_size,
            shuffle,
            clf_transform=clf_transform,
            attr_transform=attr_transform
        )
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