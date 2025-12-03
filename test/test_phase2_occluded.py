"""
Simple test for Phase 2 occluded image generation.
Tests basic functionality: path generation and structure.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.file_manager import FileManager
from config import BASE_DIR


def test_occluded_image_structure():
    """Test occluded image folder structure."""
    file_manager = FileManager(BASE_DIR)
    
    # Test path generation
    occluded_path = file_manager.get_occluded_image_path(
        "imagenet", "resnet50", "mean", "grad_cam", 50, "image_00000"
    )
    
    # Check structure: results/occluded/{dataset}/{model}/{strategy}/{method}/{level}/
    parts = occluded_path.parts
    assert "results" in parts
    assert "occluded" in parts
    assert "imagenet" in parts
    assert "resnet50" in parts
    assert "mean" in parts
    assert "grad_cam" in parts
    assert "50" in parts
    
    print("✓ Occluded image structure is correct")


if __name__ == "__main__":
    test_occluded_image_structure()
    print("All tests passed!")


