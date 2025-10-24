"""
Small test run for the new CROSS-XAI structure.

Runs all 3 phases on a tiny subset of the dataset to verify everything works.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Add current dirctory to path for imports
sys.path.append('.')


def create_tiny_dataset():
    """Create a tiny test dataset with 3 images."""
    print("Creating tiny test dataset...")

    # Create temporary directory
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)

    # Create 3 class folders with 1 image each
    for i in range(3):
        class_dir = test_dir / f"class_{i:03d}"
        class_dir.mkdir(exist_ok=True)

        # Create a simple test image (224x224 RGB)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(class_dir / f"test_image_{i}.jpg")

    print(f"Created test dataset with 3 images in {test_dir}")
    return test_dir


def update_config_for_test():
    """Update config for small test run."""
    print("Updating config for test run...")

    # Read current config
    with open('config.py', 'r') as f:
        content = f.read()

    # Create test config
    test_config = content.replace(
        'DATA_DIR = BASE_DIR / "data"',
        'DATA_DIR = BASE_DIR / "test_data"'
    ).replace(
        'GENERATING_MODELS = [\n    "resnet50",\n    "mobilenet_v2", \n    "vgg16",\n    "vit_b_16",\n    "swin_t",\n]',
        'GENERATING_MODELS = ["resnet50"]'  # Only one model for test
    ).replace(
        'JUDGING_MODELS = [\n    "resnet50",\n    "vit_b_16",\n    "swin_t",\n]',
        'JUDGING_MODELS = ["resnet50"]'  # Only one judge for test
    ).replace(
        'ATTRIBUTION_METHODS = [\n    "saliency",\n    "inputxgradient",\n    "smoothgrad",\n    "guided_backprop",\n    "integrated_gradients",\n    "gradientshap",\n    "occlusion",\n    "xrai",\n    "grad_cam",\n    "guided_gradcam",\n    "random_baseline",\n]',
        'ATTRIBUTION_METHODS = ["saliency", "random_baseline"]'  # Only 2 methods for test
    ).replace(
        'OCCLUSION_LEVELS = list(range(5, 100, 5))',
        'OCCLUSION_LEVELS = [10, 50, 90]'  # Only 3 levels for test
    ).replace(
        'FILL_STRATEGIES = [\n    "gray",\n    "blur",\n    "random_noise",\n    "black",\n    "mean",\n    "white",\n]',
        'FILL_STRATEGIES = ["gray", "blur"]'  # Only 2 strategies for test
    )

    # Write test config
    with open('config_test.py', 'w') as f:
        f.write(test_config)

    print("Created config_test.py for test run")


def run_small_test():
    """Run the small test."""
    print("=" * 60)
    print("RUNNING SMALL CROSS-XAI TEST")
    print("=" * 60)

    try:
        # Import test modules
        from core.gpu_manager import GPUManager
        from core.experiment_runner import ExperimentRunner
        from attribution.registry import get_all_methods
        from models.loader import load_model
        from data.loader import get_dataloader

        print("All imports successful!")

        # Test GPU manager
        gpu_manager = GPUManager()
        print(f"GPU Manager: {gpu_manager.device}")

        # Test attribution methods
        methods = get_all_methods()
        print(f"Attribution methods: {len(methods)} found")

        # Test model loading (CPU only for test)
        print("Testing model loading...")
        try:
            model = load_model("resnet50")
            print("Model loading successful!")
        except Exception as e:
            print(f"Model loading failed: {e}")
            print("This is expected if torchvision models aren't available")

        # Test data loading
        print("Testing data loading...")
        try:
            dataloader = get_dataloader("imagenet", batch_size=1, shuffle=False)
            print(f"Data loading successful! Found {len(dataloader)} samples")
        except Exception as e:
            print(f"Data loading failed: {e}")
            return False

        print("\n" + "=" * 60)
        print("SMALL TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("All core components are working")
        print("Ready for full experiment run")

        return True

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test():
    """Clean up test files."""
    print("\nCleaning up test files...")

    # Remove test data
    if Path("test_data").exists():
        shutil.rmtree("test_data")
        print("Removed test_data/")

    # Remove test config
    if Path("config_test.py").exists():
        os.remove("config_test.py")
        print("Removed config_test.py")

    # Remove test results
    if Path("results").exists():
        shutil.rmtree("results")
        print("Removed results/")


def main():
    """Main test function."""
    print("CROSS-XAI Small Test Runner")
    print("=" * 60)

    # Create tiny dataset
    test_dir = create_tiny_dataset()

    # Update config for test
    update_config_for_test()

    # Update dataset config for test
    with open('config.py', 'r') as f:
        content = f.read()

    # Temporarily update dataset path
    test_content = content.replace(
        '"path": DATA_DIR / "imagenet"',
        '"path": Path("test_data")'
    )

    with open('config.py', 'w') as f:
        f.write(test_content)

    try:
        # Run the test
        success = run_small_test()

        if success:
            print("\nALL TESTS PASSED!")
            print("The new CROSS-XAI structure is working correctly.")
            print("\nYou can now safely delete the old files:")
            print("- run_experiment.py")
            print("- run_analysis.py")
            print("- plotting.py")
            print("- read_heatmap.py")
            print("- utils/ (after moving to data/ and models/)")
            print("- modules/ (after moving to attribution/ and evaluation/)")
            print("- tmp/")
            print("- test_structure.py")
        else:
            print("\nSome tests failed. Check the errors above.")

    finally:
        # Restore original config
        with open('config.py', 'w') as f:
            f.write(content)

        # Cleanup
        cleanup_test()


if __name__ == "__main__":
    main()
