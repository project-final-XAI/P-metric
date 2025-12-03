"""
Configuration for SSMS vs P-Metric comparison experiment.

Reuses settings from root config.py and adds experiment-specific parameters.
"""

import sys
from pathlib import Path

# Add parent directory to path to import root config
sys.path.insert(0, str(Path(__file__).parent.parent))
import config as root_config


class ExperimentConfig:
    """Configuration for SSMS vs P-Metric experiment."""

    def __init__(self, quick_mode=False, num_images=None):
        """
        Initialize experiment configuration.
        
        Args:
            quick_mode: If True, use minimal settings for quick test (20 images)
            num_images: Override number of images (None = use default)
        """
        # Reuse root config settings
        self.base_dir = root_config.BASE_DIR
        self.device = root_config.DEVICE
        self.dataset_name = root_config.DATASET_NAME
        self.data_dir = root_config.DATA_DIR
        self.heatmap_dir = root_config.HEATMAP_DIR

        # Experiment-specific parameters
        if quick_mode:
            self.num_images = 20
        elif num_images is not None:
            self.num_images = num_images
        else:
            self.num_images = 1000

        # Models: use first generating model from config (or resnet50 as default)
        self.generating_models = root_config.GENERATING_MODELS[0]

        # Explainers: use first 2 attribution methods (or defaults)
        available_methods = root_config.ATTRIBUTION_METHODS
        if not available_methods:
            available_methods = ['grad_cam', 'integrated_gradients']
        self.explainers = available_methods

        # Judge models: filter out LLM models for speed (only PyTorch models)
        self.judge_models = [m for m in root_config.JUDGING_MODELS if not m.startswith('llama')]
        if not self.judge_models:
            # Fallback to defaults if all were LLMs
            self.judge_models = ['resnet50', 'mobilenet_v2']

        # Occlusion parameters for P-Metric
        # Reduced from 19 levels (5,10,...,95) to 5 levels for speed
        # Original: list(range(5, 100, 5)) = 19 levels
        # New: [0, 25, 50, 75, 95] = 5 levels (3.8x faster!)
        self.occlusion_percents = [0, 25, 50, 75, 95]
        
        # Option to skip P-Metric entirely (only compute SSMS) - much faster!
        self.skip_pmetric = False  # Set to True to skip P-Metric and only compute SSMS
        self.fill_strategy = 'mean'  # Use mean fill strategy (from config default)

        # SSMS parameters
        self.alpha_max = 10.0
        self.eps = 1e-8

        # Output directories
        self.experiment_dir = Path(__file__).parent
        self.plots_dir = self.experiment_dir / "plots"
        self.visuals_dir = self.experiment_dir / "visuals"
        self.results_dir = self.experiment_dir

        # Ensure output directories exist
        self.plots_dir.mkdir(exist_ok=True)
        self.visuals_dir.mkdir(exist_ok=True)

