"""
Run all 4 phases of the pipeline.

This is the main entry point for running the complete experiment.
"""

import logging
import sys
from pathlib import Path
import torch
# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.phase1_runner import Phase1Runner
from core.phase2_runner import Phase2Runner
from core.phase3_runner import Phase3Runner
from core.phase4_runner import Phase4Runner

def main():
    """Run all 4 phases."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize shared resources
    gpu_manager = GPUManager()
    gpu_manager.print_info()
    
    file_manager = FileManager(config.BASE_DIR)
    file_manager.ensure_dir_exists(file_manager.heatmap_dir)
    file_manager.ensure_dir_exists(file_manager.results_dir)
    file_manager.ensure_dir_exists(file_manager.analysis_dir)
    
    model_cache = {}

    # Run phases
    logging.info("=" * 60)
    logging.info("Starting 4-Phase Pipeline")
    logging.info("=" * 60)

    from data.loader import get_dataset_handler
    from models.loader import get_model_provider
    dataset_handler = get_dataset_handler(config.DATASET_NAME)
    model_provider = get_model_provider(config.DATASET_NAME)
    phase1 = Phase1Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase1.run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 1")

    phase2 = Phase2Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase2.run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 2")

    phase3 = Phase3Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase3.run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 3")

    phase4 = Phase4Runner(config, gpu_manager, file_manager, dataset_handler, model_provider)
    phase4.run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 4")
    
    logging.info("=" * 60)
    logging.info("All phases complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()


