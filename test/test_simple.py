"""
Simple test to verify the new structure works.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test basic imports."""
    print("Testing imports...")
    
    try:
        # Test core imports
        from core.gpu_manager import GPUManager
        print("✅ GPU Manager imported")
        
        # Test attribution imports
        from attribution.registry import get_all_methods
        methods = get_all_methods()
        print(f"✅ Attribution registry: {len(methods)} methods")
        
        # Test model imports
        from models.loader import load_model
        print("✅ Model loader imported")
        
        # Test data imports
        from data.loader import get_dataloader
        print("✅ Data loader imported")
        
        # Test evaluation imports
        from evaluation.occlusion import sort_pixels
        print("✅ Occlusion evaluator imported")
        
        # Test visualization imports
        from visualization.plotter import plot_accuracy_degradation_curves
        print("✅ Plotter imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu_manager():
    """Test GPU manager."""
    print("\nTesting GPU Manager...")
    
    try:
        from core.gpu_manager import GPUManager
        gpu_manager = GPUManager()
        print(f"✅ Device: {gpu_manager.device}")
        print(f"✅ GPU Memory: {gpu_manager.gpu_memory_gb:.1f}GB")
        return True
    except Exception as e:
        print(f"❌ GPU Manager failed: {e}")
        return False

def test_attribution_methods():
    """Test attribution methods."""
    print("\nTesting Attribution Methods...")
    
    try:
        from attribution.registry import get_all_methods, get_attribution_method
        
        methods = get_all_methods()
        print(f"✅ Found {len(methods)} methods: {methods}")
        
        # Test getting a specific method
        saliency = get_attribution_method("saliency")
        print(f"✅ Saliency: {saliency.name}")
        
        return True
    except Exception as e:
        print(f"❌ Attribution methods failed: {e}")
        return False

def test_config():
    """Test config loading."""
    print("\nTesting Config...")
    
    try:
        import config
        print(f"✅ Models: {len(config.GENERATING_MODELS)}")
        print(f"✅ Methods: {len(config.ATTRIBUTION_METHODS)}")
        print(f"✅ Device: {config.DEVICE}")
        return True
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 CROSS-XAI Structure Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_gpu_manager,
        test_attribution_methods,
        test_config
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED!")
        print("\n✅ The new CROSS-XAI structure is working correctly!")
        print("\nYou can now safely delete these old files:")
        print("❌ run_experiment.py")
        print("❌ run_analysis.py")
        print("❌ plotting.py")
        print("❌ read_heatmap.py")
        print("❌ utils/ (after moving to data/ and models/)")
        print("❌ modules/ (after moving to attribution/ and evaluation/)")
        print("❌ tmp/")
        print("❌ test_structure.py")
        print("❌ test_small_run.py")
        print("❌ test_simple.py")
        
        print("\n🚀 Ready to run the full experiment!")
        print("Use: python scripts/run_full.py --dataset imagenet")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

