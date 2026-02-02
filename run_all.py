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
from models.loader import load_model
from evaluation.judging.binary_llm_judge import BinaryLLMJudge
from evaluation.judging.cosine_llm_judge import CosineSimilarityLLMJudge
from evaluation.judging.classid_llm_judge import ClassIdLLMJudge


def get_cached_model(name, model_cache, dataset_name):
    """Load model with caching. Supports both PyTorch models and JudgingModel instances."""
    if name not in model_cache:
        if name.endswith('-binary'):
            logging.info(f"Loading Binary LLM judge: {name}")
            model_cache[name] = BinaryLLMJudge(
                model_name=name,
                dataset_name=dataset_name,
                temperature=0.0
            )
        elif name.endswith('-cosine'):
            logging.info(f"Loading Cosine Similarity LLM judge: {name}")
            model_cache[name] = CosineSimilarityLLMJudge(
                model_name=name,
                dataset_name=dataset_name,
                temperature=0.1,
                similarity_threshold=0.8,
                embedding_model="nomic-embed-text"
            )
        elif name.endswith('-classid'):
            logging.info(f"Loading ClassId LLM judge: {name}")
            model_cache[name] = ClassIdLLMJudge(
                model_name=name,
                dataset_name=dataset_name,
                temperature=0.0
            )
        else:
            logging.info(f"Loading PyTorch model: {name}")
            model_cache[name] = load_model(name)
    return model_cache[name]


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
    
    def get_cached_model_func(name):
        return get_cached_model(name, model_cache, config.DATASET_NAME)
    
    # Run phases
    logging.info("=" * 60)
    logging.info("Starting 4-Phase Pipeline")
    logging.info("=" * 60)
    
    phase1 = Phase1Runner(config, gpu_manager, file_manager, model_cache)
    phase1.run(get_cached_model_func)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 1")
    
    phase2 = Phase2Runner(config, gpu_manager, file_manager, model_cache)
    phase2.run(get_cached_model_func)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 2")

    phase3 = Phase3Runner(config, gpu_manager, file_manager, model_cache)
    phase3.run(get_cached_model_func)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 3")

    phase4 = Phase4Runner(config, file_manager)
    phase4.run()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logging.info("Cleared GPU cache after Phase 4")
    
    logging.info("=" * 60)
    logging.info("All phases complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()


