"""
Basic structure test - works without heavy dependencies.
"""

import os
import sys
from pathlib import Path

def test_directory_structure():
    """Test that all directories exist."""
    print("Testing directory structure...")
    
    required_dirs = [
        "core",
        "attribution", 
        "models",
        "evaluation",
        "data",
        "visualization",
        "scripts"
    ]
    
    missing = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing.append(dir_name)
        else:
            print(f"✅ {dir_name}/")
    
    if missing:
        print(f"❌ Missing directories: {missing}")
        return False
    
    print("✅ All directories exist!")
    return True

def test_key_files():
    """Test that key files exist."""
    print("\nTesting key files...")
    
    required_files = [
        "core/experiment_runner.py",
        "core/gpu_manager.py",
        "attribution/registry.py",
        "attribution/base.py",
        "models/loader.py",
        "evaluation/occlusion.py",
        "data/loader.py",
        "visualization/plotter.py",
        "scripts/run_full.py",
        "scripts/run_phase1.py",
        "scripts/run_phase2.py", 
        "scripts/run_phase3.py",
        "config.py",
        "README.md"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing:
        print(f"❌ Missing files: {missing}")
        return False
    
    print("✅ All key files exist!")
    return True

def test_config_structure():
    """Test config.py structure."""
    print("\nTesting config.py...")
    
    try:
        with open("config.py", "r") as f:
            content = f.read()
        
        required_configs = [
            "GENERATING_MODELS",
            "JUDGING_MODELS", 
            "ATTRIBUTION_METHODS",
            "OCCLUSION_LEVELS",
            "FILL_STRATEGIES",
            "HEATMAP_DIR",
            "RESULTS_DIR",
            "ANALYSIS_DIR"
        ]
        
        missing = []
        for config in required_configs:
            if config not in content:
                missing.append(config)
            else:
                print(f"✅ {config}")
        
        if missing:
            print(f"❌ Missing configs: {missing}")
            return False
        
        print("✅ Config structure looks good!")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_import_structure():
    """Test basic import structure (without heavy dependencies)."""
    print("\nTesting import structure...")
    
    try:
        # Test that __init__.py files exist
        init_files = [
            "core/__init__.py",
            "attribution/__init__.py", 
            "models/__init__.py",
            "evaluation/__init__.py",
            "data/__init__.py",
            "visualization/__init__.py",
            "scripts/__init__.py"
        ]
        
        missing = []
        for init_file in init_files:
            if not Path(init_file).exists():
                missing.append(init_file)
            else:
                print(f"✅ {init_file}")
        
        if missing:
            print(f"❌ Missing __init__.py files: {missing}")
            return False
        
        print("✅ All __init__.py files exist!")
        return True
        
    except Exception as e:
        print(f"❌ Import structure test failed: {e}")
        return False

def main():
    """Run all basic tests."""
    print("🧪 CROSS-XAI Basic Structure Test")
    print("=" * 50)
    
    tests = [
        test_directory_structure,
        test_key_files,
        test_config_structure,
        test_import_structure
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Tests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 ALL BASIC TESTS PASSED!")
        print("\n✅ The new CROSS-XAI structure is correctly organized!")
        print("\n📁 Directory structure:")
        print("├── core/           # Experiment orchestration")
        print("├── attribution/    # 11 XAI methods")
        print("├── models/         # Model loading & architectures")
        print("├── evaluation/     # Occlusion & metrics")
        print("├── data/           # Dataset loading")
        print("├── visualization/  # Plotting utilities")
        print("└── scripts/        # Entry points")
        
        print("\n🗑️  Files you can now safely delete:")
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
        print("❌ test_basic.py")
        
        print("\n🚀 Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full experiment: python scripts/run_full.py --dataset imagenet")
        print("3. Or run individual phases: python scripts/run_phase1.py --dataset imagenet")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()

