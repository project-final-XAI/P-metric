"""
Single file for running quick test of the entire system (3 phases).

Runs all phases with limited data for faster testing.

Usage:
    python test/run_quick_test.py
    python test/run_quick_test.py --max-images 50
    python test/run_quick_test.py --phase 1
"""

import sys
from pathlib import Path
import logging
import torch
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.phase2_runner import Phase2Runner
from core.phase3_runner import Phase3Runner
from core.phase4_runner import Phase4Runner
from models.loader import get_model_provider
from evaluation.judging.registry import register_judging_model, get_judging_model
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.classid_llm_judge import ClassIdLLMJudge
from data.loader import get_dataset_handler
from attribution.registry import get_attribution_method
from evaluation.occlusion import sort_pixels

# Setup logging with better formatting
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

# ===========================
# Test Configuration
# ===========================
MAX_IMAGES = 10  # Maximum number of images to process (set to 10 for quick test)
DATASET_NAME = "imagenet"
GENERATING_MODELS = ["resnet50"]
JUDGING_MODELS = ["resnet50"]
ATTRIBUTION_METHODS = [
    "grad_cam", "guided_gradcam", "expected_gradcam",
    "dino", "u2net", "u2net+dino",
    "dino_pca_unet_match", "dinov2_pca_gaussian", "dinov2_pca_attention"
]
OCCLUSION_LEVELS = [0, 50] # list(range(10, 100, 10))  # [10, 20, 30, ..., 90]
FILL_STRATEGIES = ["mean"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QuickTestConfig:
    """Simple config object for quick test."""
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    HEATMAP_DIR = BASE_DIR / "results" / "heatmaps"
    RESULTS_DIR = BASE_DIR / "results" / "evaluation"
    ANALYSIS_DIR = BASE_DIR / "results" / "analysis"
    
    MAX_WORKERS = 8
    DEVICE = DEVICE
    HEATMAP_BATCH_SIZE = 12
    USE_FP16_INFERENCE = True
    USE_TORCH_COMPILE = True
    PROGRESS_AUTO_SAVE_INTERVAL = 50
    PROGRESS_AUTO_SAVE_TIME = 300
    
    DATASET_CONFIG = {
        "imagenet": {"path": DATA_DIR / "imagenet", "num_classes": 1000},
        "SIPaKMeD": {"path": DATA_DIR / "SIPaKMeD", "num_classes": 5},
        "SIPaKMeD_cropped": {"path": DATA_DIR / "SIPaKMeD_cropped", "num_classes": 5}
    }
    
    DATASET_NAME = DATASET_NAME
    GENERATING_MODELS = GENERATING_MODELS
    JUDGING_MODELS = JUDGING_MODELS
    ATTRIBUTION_METHODS = ATTRIBUTION_METHODS
    OCCLUSION_LEVELS = OCCLUSION_LEVELS
    FILL_STRATEGIES = FILL_STRATEGIES


def print_section_header(title: str):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(" " * ((70 - len(title)) // 2) + title)
    print("="*70)


def print_config_info(config, max_images):
    """Print configuration information in a formatted way."""
    print("\nConfiguration:")
    print(f"  {'Dataset:':<25} {config.DATASET_NAME}")
    print(f"  {'Max Images:':<25} {max_images}")
    print(f"  {'Generating Models:':<25} {', '.join(config.GENERATING_MODELS)}")
    print(f"  {'Judging Models:':<25} {', '.join(config.JUDGING_MODELS)}")
    print(f"  {'Attribution Methods:':<25} {', '.join(config.ATTRIBUTION_METHODS)}")
    print(f"  {'Occlusion Levels:':<25} {config.OCCLUSION_LEVELS}")
    print(f"  {'Fill Strategies:':<25} {', '.join(config.FILL_STRATEGIES)}")


from core.phase1_runner import Phase1Runner

def run_phase1_limited(config, gpu_manager, file_manager, model_cache, max_images, dataset_handler, model_provider):
    """Run Phase 1 with image limit."""
    print_section_header("PHASE 1: HEATMAP GENERATION")
    runner = Phase1Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    
    # Patch _load_image_label_map to restrict images
    original_load = runner._load_image_label_map
    def limited_load():
        full_map = original_load()
        return dict(list(full_map.items())[:max_images])
    
    runner._load_image_label_map = limited_load
    runner.run()


def run_phase2_limited(config, gpu_manager, file_manager, dataset_handler, model_provider):
    """Run Phase 2."""
    print_section_header("PHASE 2: OCCLUSION EVALUATION")
    phase2_runner = Phase2Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase2_runner.run()


def run_phase3(config, gpu_manager, file_manager, dataset_handler, model_provider):
    """Run Phase 3."""
    print_section_header("PHASE 3: SUPER-FAST EVALUATION")
    phase3_runner = Phase3Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase3_runner.run()


def run_phase4(config, gpu_manager, file_manager, dataset_handler, model_provider):
    """Run Phase 4."""
    print_section_header("PHASE 4: ANALYSIS AND VISUALIZATION")
    phase4_runner = Phase4Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase4_runner.run()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run quick test with limited data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test/run_quick_test.py
  python test/run_quick_test.py --max-images 50
  python test/run_quick_test.py --phase 1
        """
    )
    parser.add_argument('--max-images', type=int, default=MAX_IMAGES,
                       help=f'Maximum number of images (default: {MAX_IMAGES})')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4], default=None,
                       help='Run only specific phase (1, 2, 3, or 4)')
    
    args = parser.parse_args()
    
    config = QuickTestConfig()
    
    print_section_header("QUICK TEST - LIMITED DATA EXPERIMENT")
    print_config_info(config, args.max_images)
    print("="*70)
    
    # Initialize resources
    logging.info("Initializing resources...")
    gpu_manager = GPUManager()
    gpu_manager.print_info()
    file_manager = FileManager(config.BASE_DIR)
    file_manager.ensure_dir_exists(file_manager.heatmap_dir)
    file_manager.ensure_dir_exists(file_manager.results_dir)
    file_manager.ensure_dir_exists(file_manager.analysis_dir)
    
    dataset_handler = get_dataset_handler(config.DATASET_NAME)
    model_provider = get_model_provider(config.DATASET_NAME)
    
    model_cache = {}
    
    # Register LLM judges
    try:
        logging.info("Registering LLM judges...")
        # Create factory function for binary LLM judge
        def binary_llm_factory(model_name: str):
            return BinaryLLMJudge(
                model_name=model_name,
                dataset_name=config.DATASET_NAME,
                temperature=0.0
            )
        # Create factory function for classid LLM judge
        def classid_llm_factory(model_name: str):
            return ClassIdLLMJudge(
                model_name=model_name,
                dataset_name=config.DATASET_NAME,
                temperature=0.0
            )
        register_judging_model("llama3.2-vision-binary", binary_llm_factory)
        register_judging_model("llama3.2-vision-classid", classid_llm_factory)
        logging.info("LLM judges registered successfully")
    except Exception as e:
        logging.warning(f"Failed to register LLM judges: {e}")
    
    # Run phases
    try:
        if args.phase is None or args.phase == 1:
            run_phase1_limited(config, gpu_manager, file_manager, model_cache, args.max_images, dataset_handler, model_provider)
        
        if args.phase is None or args.phase == 2:
            run_phase2_limited(config, gpu_manager, file_manager, dataset_handler, model_provider)
        
        if args.phase is None or args.phase == 3:
            run_phase3(config, gpu_manager, file_manager, dataset_handler, model_provider)
            
        if args.phase is None or args.phase == 4:
            run_phase4(config, gpu_manager, file_manager, dataset_handler, model_provider)
        
        print_section_header("QUICK TEST COMPLETE")
        logging.info("All phases completed successfully!")
        print("="*70 + "\n")
    except Exception as e:
        logging.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

