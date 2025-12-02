"""
Simple test for Phase 3 image loading.
Tests basic functionality: loading images and checking paths.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_manager import FileManager
from config import BASE_DIR


def test_image_loading_paths():
    """Test image loading path methods."""
    file_manager = FileManager(BASE_DIR)
    
    # Test scanning occluded images
    occluded_dir = file_manager.get_occluded_dir("imagenet")
    assert "occluded" in str(occluded_dir)
    
    # Test checking if image exists (will be False if not generated yet)
    exists = file_manager.check_occluded_image_exists(
        "imagenet", "resnet50", "mean", "grad_cam", 50, "image_00000"
    )
    # This is fine - just checking the method works
    assert isinstance(exists, bool)
    
    print("✓ Image loading methods work correctly")


if __name__ == "__main__":
    test_image_loading_paths()
    print("All tests passed!")

