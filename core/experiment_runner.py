"""
Main experiment orchestration for CROSS-XAI evaluation.

Coordinates the 3-phase pipeline by delegating to phase-specific runners:
1. Phase 1: Heatmap generation (attribution maps)
2. Phase 2: Occlusion-based evaluation
3. Phase 3: Metrics calculation and visualization
"""

import os
import logging
import torch

from core.gpu_manager import GPUManager
from core.file_manager import FileManager
from core.phase1_runner import Phase1Runner
from core.phase2_runner import Phase2Runner
from core.phase3_runner import Phase3Runner
from models.loader import load_model
from evaluation.judging.registry import get_judging_model, register_judging_model, create_llamavision_judge_factory
from evaluation.judging.base import JudgingModel

# Setup logging (separate from tqdm stdout)
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)


class ExperimentRunner:
    """
    Main experiment orchestrator.
    
    Coordinates the 3-phase pipeline by delegating to phase-specific runners.
    Maintains shared resources like GPU manager, file manager, and model cache.
    """
    
    def __init__(self, config):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration object with experiment parameters
        """
        self.config = config
        
        # Normalize deprecated allocator env var to silence warnings
        self._normalize_allocator_env()
        
        # Initialize shared resources
        self.gpu_manager = GPUManager()
        self.gpu_manager.print_info()
        
        self.file_manager = FileManager(config.BASE_DIR)
        self._ensure_directories()
        
        # Model cache to avoid reloading (supports both PyTorch models and JudgingModel instances)
        self._model_cache = {}
        
        # Register LLM judges if needed
        self._register_judging_models()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize phase runners
        self.phase1_runner = Phase1Runner(
            config, self.gpu_manager, self.file_manager, self._model_cache
        )
        self.phase2_runner = Phase2Runner(
            config, self.gpu_manager, self.file_manager, self._model_cache
        )
        self.phase3_runner = Phase3Runner(
            config, self.file_manager
        )
    
    def _normalize_allocator_env(self):
        """Normalize deprecated PYTORCH_CUDA_ALLOC_CONF environment variable."""
        try:
            if "PYTORCH_CUDA_ALLOC_CONF" in os.environ and "PYTORCH_ALLOC_CONF" not in os.environ:
                os.environ["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
                del os.environ["PYTORCH_CUDA_ALLOC_CONF"]
        except Exception:
            pass
    
    def _ensure_directories(self):
        """Create base directories if they don't exist."""
        self.file_manager.ensure_dir_exists(self.file_manager.heatmap_dir)
        self.file_manager.ensure_dir_exists(self.file_manager.results_dir)
        self.file_manager.ensure_dir_exists(self.file_manager.analysis_dir)
    
    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.config.GENERATING_MODELS:
            raise ValueError("GENERATING_MODELS cannot be empty")
        if not self.config.JUDGING_MODELS:
            raise ValueError("JUDGING_MODELS cannot be empty")
        if not self.config.ATTRIBUTION_METHODS:
            raise ValueError("ATTRIBUTION_METHODS cannot be empty")
        if not self.config.OCCLUSION_LEVELS:
            raise ValueError("OCCLUSION_LEVELS cannot be empty")
        if not self.config.FILL_STRATEGIES:
            raise ValueError("FILL_STRATEGIES cannot be empty")
    
    def _register_judging_models(self):
        """Register judging model factories for LLM judges."""
        try:
            dataset_name = getattr(self.config, 'DATASET_NAME', 'imagenet')
            factory = create_llamavision_judge_factory(dataset_name=dataset_name)
            # Register common Ollama model names
            ollama_models = ['llama3.2-vision', 'llama-vision', 'llama3.2:latest']
            for model_name in ollama_models:
                register_judging_model(model_name, factory)
                logging.debug(f"Registered LlamaVision judge factory for {model_name}")
        except Exception as e:
            logging.warning(f"Failed to register LLM judges: {e}")
    
    def _get_cached_model(self, model_name: str):
        """
        Load model with caching. Supports both PyTorch models and JudgingModel instances.
        
        Checks the judging model registry first. If not found, loads as PyTorch model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Model instance (PyTorch model or JudgingModel)
        """
        if model_name not in self._model_cache:
            # Check if it's a registered judging model (LLM judge)
            judging_model = get_judging_model(model_name)
            if judging_model is not None:
                logging.info(f"Loading judging model from registry: {model_name}")
                self._model_cache[model_name] = judging_model
            else:
                # Load as PyTorch model
                logging.info(f"Loading PyTorch model: {model_name}")
                model = load_model(model_name)
                model = self._maybe_compile_model(model, model_name)
                self._model_cache[model_name] = model
        return self._model_cache[model_name]
    
    def _maybe_compile_model(self, model: torch.nn.Module, model_name: str) -> torch.nn.Module:
        """
        Model compilation hook (currently disabled for maximum compatibility).
        
        Args:
            model: PyTorch model
            model_name: Name of the model
            
        Returns:
            Model (unchanged, compilation disabled)
        """
        return model
    
    def run_phase_1(self):
        """Run Phase 1: Generate heatmaps for all model-method-image combinations."""
        self.phase1_runner.run(self._get_cached_model)
    
    def run_phase_2(self):
        """Run Phase 2: Evaluate heatmaps with occlusion."""
        self.phase2_runner.run(self._get_cached_model)
    
    def run_phase_3(self):
        """Run Phase 3: Calculate metrics and generate visualizations."""
        self.phase3_runner.run()
