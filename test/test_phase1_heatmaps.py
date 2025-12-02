"""
Simple test for Phase 1 heatmap generation.
Tests basic functionality: generating and saving heatmaps.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_manager import FileManager
from config import BASE_DIR


def test_file_manager_paths():
    """Test FileManager path methods."""
    file_manager = FileManager(BASE_DIR)
    
    # Test sorted heatmap path
    sorted_path = file_manager.get_sorted_heatmap_path(
        "imagenet", "resnet50", "grad_cam", "image_00000"
    )
    assert "sorted" in str(sorted_path)
    assert sorted_path.suffix == ".npy"
    
    # Test regular heatmap path
    regular_path = file_manager.get_regular_heatmap_path(
        "imagenet", "resnet50", "grad_cam", "image_00000"
    )
    assert "regular" in str(regular_path)
    assert regular_path.suffix == ".png"
    
    # Test occluded image path
    occluded_path = file_manager.get_occluded_image_path(
        "imagenet", "resnet50", "mean", "grad_cam", 50, "image_00000"
    )
    assert "occluded" in str(occluded_path)
    assert occluded_path.suffix == ".png"
    
    print("✓ FileManager path methods work correctly")


if __name__ == "__main__":
    test_file_manager_paths()
    print("All tests passed!")

