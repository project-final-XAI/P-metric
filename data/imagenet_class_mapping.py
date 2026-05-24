"""
ImageNet class mapping utility.

Resolves integer labels directly to human-readable names,
leveraging torchvision's built-in, pre-sorted metadata.
"""

import logging
from typing import Optional
from torchvision.models import ResNet50_Weights

# Extract the static list from torchvision (no downloading happens here)
_IMAGENET_CATEGORIES = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def get_imagenet_category_name(label: int) -> Optional[str]:
    """
    Directly converts a DataLoader integer label (0-999) to a clean string.

    Args:
        label: Integer class index from PyTorch DataLoader.

    Returns:
        Human-readable name, or None if label is out of bounds.
    """
    try:
        if 0 <= label < len(_IMAGENET_CATEGORIES):
            return _IMAGENET_CATEGORIES[label]

        logging.warning(f"Label {label} is out of bounds for ImageNet-1K.")
        return None
    except Exception as e:
        logging.error(f"Error resolving ImageNet label {label}: {e}")
        return None


if __name__ == "__main__":
    # Test the mapping
    logging.basicConfig(level=logging.INFO)

    print("Testing ImageNet Integer -> LLM Name mapping:\n")
    for test_label in [0, 1, 2, 999]:
        name = get_imagenet_category_name(test_label)
        print(f"Label {test_label:3d} -> {name}")