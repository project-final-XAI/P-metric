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
from models.loader import load_model
from evaluation.judging.registry import register_judging_model, get_judging_model
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.classid_llm_judge import ClassIdLLMJudge
from data.loader import get_dataloader
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
JUDGING_MODELS = ["llama3.2-vision-binary"]
ATTRIBUTION_METHODS = ["gradcam"] #["inputxgradient", "grad_cam", "random_baseline"]
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


def run_phase1_limited(config, gpu_manager, file_manager, model_cache, max_images):
    """Run Phase 1 with image limit."""
    print_section_header("PHASE 1: HEATMAP GENERATION")
    
    dataset_name = config.DATASET_NAME
    heatmap_dir = file_manager.get_heatmap_dir(dataset_name)
    file_manager.ensure_dir_exists(heatmap_dir)
    
    logging.info("Loading dataset with image limit...")
    dataloader = get_dataloader(dataset_name, batch_size=32, shuffle=False)
    image_label_map = {}
    global_idx = 0
    
    for batch_images, batch_labels in dataloader:
        if max_images and global_idx >= max_images:
            break
        for img, lbl in zip(batch_images, batch_labels):
            if max_images and global_idx >= max_images:
                break
            image_label_map[f"image_{global_idx:05d}"] = (img, lbl.item())
            global_idx += 1
    
    logging.info(f"Processing {len(image_label_map)} images (limited from dataset)")
    
    # Process each model-method combination
    total_combinations = len(config.GENERATING_MODELS) * len(config.ATTRIBUTION_METHODS)
    with tqdm(total=total_combinations, desc="Phase 1 Progress", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for model_idx, model_name in enumerate(config.GENERATING_MODELS, 1):
            if model_name not in model_cache:
                logging.info(f"Loading model: {model_name}")
                model_cache[model_name] = load_model(model_name)
            model = model_cache[model_name]
            
            for method_idx, method_name in enumerate(config.ATTRIBUTION_METHODS, 1):
                pbar.set_description(
                    f"[{model_idx}/{len(config.GENERATING_MODELS)}] {model_name[:12]} | "
                    f"[{method_idx}/{len(config.ATTRIBUTION_METHODS)}] {method_name[:15]}"
                )
                try:
                    method = get_attribution_method(method_name)
                    batch_size = gpu_manager.get_batch_size(method_name)
                    
                    # Collect images to process
                    images_to_process = []
                    image_ids = []
                    labels = []
                    
                    for img_id, (img, label) in list(image_label_map.items()):
                        sorted_path = file_manager.get_heatmap_path(
                            dataset_name, model_name, method_name, img_id, sorted=True
                        )
                        if not sorted_path.exists():
                            images_to_process.append(img)
                            image_ids.append(img_id)
                            labels.append(label)
                    
                    if images_to_process:
                        logging.debug(f"Processing {len(images_to_process)} images for {model_name}-{method_name}")
                        # Process in batches
                        for i in range(0, len(images_to_process), batch_size):
                            end_idx = min(i + batch_size, len(images_to_process))
                            batch_images = torch.stack(images_to_process[i:end_idx]).to(config.DEVICE)
                            batch_labels = torch.tensor(labels[i:end_idx]).to(config.DEVICE)
                            
                            # Generate heatmaps
                            if config.DEVICE == "cuda":
                                with torch.amp.autocast(config.DEVICE):
                                    heatmaps = method.compute(model, batch_images, batch_labels)
                            else:
                                heatmaps = method.compute(model, batch_images, batch_labels)
                            
                            # Save sorted pixel indices
                            if heatmaps is not None:
                                for j, heatmap in enumerate(heatmaps):
                                    img_id = image_ids[i + j]
                                    heatmap_np = heatmap.cpu().numpy()
                                    sorted_indices = sort_pixels(heatmap_np)
                                    sorted_path = file_manager.get_heatmap_path(
                                        dataset_name, model_name, method_name, img_id, sorted=True
                                    )
                                    import numpy as np
                                    np.save(sorted_path, sorted_indices)
                
                except Exception as e:
                    logging.error(f"Error processing {model_name}-{method_name}: {e}")
                finally:
                    pbar.update(1)
    
    logging.info(f"Heatmaps saved to: {heatmap_dir}")


def run_phase2_limited(config, gpu_manager, file_manager, model_cache, max_images):
    """Run Phase 2 with image limit."""
    print_section_header("PHASE 2: OCCLUSION EVALUATION")
    
    # Use the existing Phase2Runner but limit images
    phase2_runner = Phase2Runner(config, gpu_manager, file_manager, model_cache)
    
    # Override _load_dataset to limit images
    original_load_dataset = phase2_runner._load_dataset
    
    def limited_load_dataset(dataset_name):
        dataloader = get_dataloader(dataset_name, batch_size=1, shuffle=False)
        image_label_map = {}
        for i, (img, lbl) in enumerate(dataloader):
            if max_images and i >= max_images:
                break
            image_label_map[f"image_{i:05d}"] = (img, lbl.item())
        logging.info(f"Processing {len(image_label_map)} images in Phase 2 (limited)")
        return image_label_map
    
    phase2_runner._load_dataset = limited_load_dataset
    
    # Override _get_heatmap_groups to filter only relevant heatmaps
    original_get_heatmap_groups = phase2_runner._get_heatmap_groups
    
    def filtered_get_heatmap_groups(dataset_name):
        """Get heatmap groups filtered to only include images from limited dataset."""
        # First get all heatmap groups
        all_groups = original_get_heatmap_groups(dataset_name)
        
        # Load limited dataset to get image IDs
        limited_image_map = limited_load_dataset(dataset_name)
        valid_image_ids = set(limited_image_map.keys())
        
        # Filter heatmap groups to only include valid image IDs
        filtered_groups = {}
        for (gen_model, method), heatmap_paths in all_groups.items():
            filtered_paths = []
            for heatmap_path in heatmap_paths:
                # Extract image ID from filename: e.g., "resnet50-grad_cam-image_00042_sorted.npy"
                parts = heatmap_path.stem.split('-')
                if len(parts) >= 3:
                    # Reconstruct image ID (e.g., "image_00042")
                    img_id = '-'.join(parts[2:]).replace('_sorted', '')
                    if img_id in valid_image_ids:
                        filtered_paths.append(heatmap_path)
            
            if filtered_paths:
                filtered_groups[(gen_model, method)] = filtered_paths
        
        logging.info(f"Filtered heatmaps: {sum(len(v) for v in filtered_groups.values())} heatmaps "
                    f"for {len(valid_image_ids)} images")
        return filtered_groups
    
    phase2_runner._get_heatmap_groups = filtered_get_heatmap_groups
    
    # Create a proper get_cached_model function that handles LLM judges
    def get_cached_model_func(model_name: str):
        """Get cached model, checking registry first for LLM judges."""
        if model_name in model_cache:
            return model_cache[model_name]
        
        # Check if it's a registered judging model (LLM judge)
        judging_model = get_judging_model(model_name)
        if judging_model is not None:
            logging.info(f"Loading judging model from registry: {model_name}")
            model_cache[model_name] = judging_model
            return judging_model
        
        # Check if it's an LLM judge by name pattern (fallback)
        dataset_name = config.DATASET_NAME
        if model_name.endswith('-binary'):
            logging.info(f"Loading Binary LLM judge: {model_name}")
            model_cache[model_name] = BinaryLLMJudge(
                model_name=model_name,
                dataset_name=dataset_name,
                temperature=0.0
            )
            return model_cache[model_name]
        elif model_name.endswith('-classid'):
            logging.info(f"Loading ClassId LLM judge: {model_name}")
            model_cache[model_name] = ClassIdLLMJudge(
                model_name=model_name,
                dataset_name=dataset_name,
                temperature=0.0
            )
            return model_cache[model_name]
        
        # Load as PyTorch model
        logging.info(f"Loading PyTorch model: {model_name}")
        model = load_model(model_name)
        model_cache[model_name] = model
        return model
    
    try:
        phase2_runner.run(get_cached_model_func)
    finally:
        phase2_runner._load_dataset = original_load_dataset
        phase2_runner._get_heatmap_groups = original_get_heatmap_groups


def run_phase3(config, file_manager):
    """Run Phase 3."""
    print_section_header("PHASE 3: ANALYSIS AND VISUALIZATION")
    
    phase3_runner = Phase3Runner(config, file_manager)
    phase3_runner.run()


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
    parser.add_argument('--phase', type=int, choices=[1, 2, 3], default=None,
                       help='Run only specific phase (1, 2, or 3)')
    
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
            run_phase1_limited(config, gpu_manager, file_manager, model_cache, args.max_images)
        
        if args.phase is None or args.phase == 2:
            run_phase2_limited(config, gpu_manager, file_manager, model_cache, args.max_images)
        
        if args.phase is None or args.phase == 3:
            run_phase3(config, file_manager)
        
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

